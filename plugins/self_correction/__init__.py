"""self-correction plugin — detects tool-level failures and injects correction nudges.

Wires three behaviours:

1. ``transform_tool_result`` hook — detects tool errors (status="error") and
   appends a compact correction nudge to the result string. The model sees
   this in the next turn and can self-correct.

2. ``pre_llm_call`` hook — detects repeating failure patterns across turns
   and injects a compact correction hint into the user message before each
   LLM call.

3. ``post_tool_call`` hook — observational only; tracks failures for pattern
   detection without modifying behavior.

Configuration:
  HERMES_SELF_CORRECTION_WINDOW (default 4) — number of turns to look back
  HERMES_SELF_CORRECTION_REPEAT_THRESHOLD (default 3) — failures before
    triggering correction
  HERMES_SELF_CORRECTION_DISABLE (default "") — set to "1" to disable

The plugin maintains a FailureTracker (module-level) that records tool
failures, detects patterns (same tool + same error category + similar
argument keys across N turns), and generates correction nudges with
category-specific suggestions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _window() -> int:
    """Number of turns to look back for pattern detection."""
    try:
        return int(os.getenv("HERMES_SELF_CORRECTION_WINDOW", "4"))
    except (ValueError, TypeError):
        return 4


def _repeat_threshold() -> int:
    """Number of failures before triggering correction."""
    try:
        return int(os.getenv("HERMES_SELF_CORRECTION_REPEAT_THRESHOLD", "3"))
    except (ValueError, TypeError):
        return 3


def _plugin_disabled() -> bool:
    """Check if the plugin is disabled via env var."""
    return os.getenv("HERMES_SELF_CORRECTION_DISABLE", "").lower() in {
        "1", "true", "yes", "on"
    }


# ---------------------------------------------------------------------------
# Error classification patterns (lowercase for case-insensitive matching)
# ---------------------------------------------------------------------------

_FILE_NOT_FOUND_PATTERNS = [
    "no such file",
    "file not found",
    "file doesn't exist",
    "does not exist",
    "no such directory",
    "directory not found",
    "path not found",
    "enoent",
]

_PERMISSION_DENIED_PATTERNS = [
    "permission denied",
    "access denied",
    "eacces",
    "unauthorized",
    "forbidden",
    "not allowed",
]

_COMMAND_FAILED_PATTERNS = [
    "command not found",
    "command failed",
    "command timed out",
    "exit code",
    "return code",
    "non-zero exit",
]

_NETWORK_ERROR_PATTERNS = [
    "connection refused",
    "connection timed out",
    "network error",
    "network unreachable",
    "dns resolution failed",
    "host not found",
    "request timed out",
    "connection reset",
]

_RESOURCE_LIMIT_PATTERNS = [
    "disk space",
    "no space left",
    "quota exceeded",
    "rate limit",
    "too many requests",
    "resource exhausted",
]

_INVALID_INPUT_PATTERNS = [
    "invalid argument",
    "invalid parameter",
    "invalid path",
    "invalid format",
    "malformed",
    "invalid json",
    "invalid syntax",
]

_TOOL_NOT_CONFIGURED_PATTERNS = [
    "not configured for use",
    "not installed",
    "not available",
    "missing dependency",
    "dependency not found",
    "requires installation",
    "setup required",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    """A single tool failure record."""
    tool_name: str
    args: Dict[str, Any]
    error_text: str
    turn_index: int


@dataclass
class FailurePattern:
    """A detected repeating failure pattern."""
    tool_name: str
    error_category: str
    repeat_count: int
    identical_args: bool
    records: List[FailureRecord]


@dataclass
class CorrectionNudge:
    """A correction nudge to inject into the conversation."""
    tool_name: str
    error_category: str
    repeat_count: int
    identical_args: bool
    suggestions: List[str]
    injection_text: str
    active: bool = True


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_tool_error(error_text: str) -> str:
    """Classify an error text into a category.

    Returns one of: file_not_found, permission_denied, command_failed,
    network_error, resource_limit, invalid_input, tool_not_configured, unknown.
    """
    text_lower = error_text.lower()

    # Check more specific patterns first
    for pattern in _COMMAND_FAILED_PATTERNS:
        if pattern in text_lower:
            return "command_failed"

    for pattern in _FILE_NOT_FOUND_PATTERNS:
        if pattern in text_lower:
            return "file_not_found"

    for pattern in _PERMISSION_DENIED_PATTERNS:
        if pattern in text_lower:
            return "permission_denied"

    for pattern in _NETWORK_ERROR_PATTERNS:
        if pattern in text_lower:
            return "network_error"

    for pattern in _RESOURCE_LIMIT_PATTERNS:
        if pattern in text_lower:
            return "resource_limit"

    for pattern in _INVALID_INPUT_PATTERNS:
        if pattern in text_lower:
            return "invalid_input"

    for pattern in _TOOL_NOT_CONFIGURED_PATTERNS:
        if pattern in text_lower:
            return "tool_not_configured"

    return "unknown"


# ---------------------------------------------------------------------------
# Argument abstraction
# ---------------------------------------------------------------------------

def abstract_arg_keys(args: Any) -> str:
    """Return a sorted key string from args dict, ignoring values.

    Used to detect when the model is trying different inputs but hitting
    the same error type (productive retry) vs. retrying the same thing
    (loop).
    """
    if not isinstance(args, dict):
        return ""
    keys = sorted(args.keys())
    return ",".join(keys)


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------

def extract_error_from_result(function_result: Any) -> str:
    """Extract the error text from a tool result string.

    Handles:
    - Plain error strings: "Error: file not found: /tmp/missing.txt"
    - Structured JSON: {"error": "file not found"}
    - Structured JSON with success=False: {"success": False, "message": "..."}
    - Nested error objects: {"error": {"code": 500, "message": "..."}}
    """
    if not function_result or not isinstance(function_result, str):
        return ""

    # Try parsing as JSON first (structured errors).
    try:
        data = json.loads(function_result)
        if isinstance(data, dict):
            # Structured error: {"error": "..."}
            if "error" in data:
                err_val = data["error"]
                # Handle nested error objects: {"error": {"code": 500, "message": "..."}}
                if isinstance(err_val, dict):
                    return str(err_val.get("message", str(err_val)))
                return str(err_val)
            # Some tools use "success": false with a message.
            if data.get("success") is False and "message" in data:
                return str(data["message"])
            # Some tools use "ok": false with a message.
            if data.get("ok") is False and "message" in data:
                return str(data["message"])
            return ""
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain error string: look for "Error:" prefix.
    if "error:" in function_result.lower():
        idx = function_result.lower().index("error:")
        return function_result[idx + 6:].strip()

    return ""


# ---------------------------------------------------------------------------
# FailureTracker
# ---------------------------------------------------------------------------

class FailureTracker:
    """Tracks tool-level failures and detects repeating patterns.

    Maintains a sliding window of recent failures. When the same tool fails
    with the same error category and similar argument keys across N turns,
    it generates a correction nudge.
    """

    def __init__(self, window: int = 4, repeat_threshold: int = 3) -> None:
        self.window = window
        self.repeat_threshold = repeat_threshold
        self._failures: List[FailureRecord] = []
        self._current_turn_index: int = 0

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    @property
    def history(self) -> List[FailureRecord]:
        return list(self._failures)

    def record_failure(
        self, tool_name: str, args: Dict[str, Any], error_text: str
    ) -> None:
        """Record a tool failure."""
        record = FailureRecord(
            tool_name=tool_name,
            args=args,
            error_text=error_text,
            turn_index=self._current_turn_index,
        )
        self._failures.append(record)
        self._prune_old_records()

    def check_for_pattern(self) -> Optional[FailurePattern]:
        """Check if there's a repeating failure pattern.

        Returns a FailurePattern if the same tool has failed with the same
        error category and similar argument keys across N turns, where N
        >= repeat_threshold.
        """
        if len(self._failures) < self.repeat_threshold:
            return None

        # Group failures by (tool_name, error_category, arg_keys)
        groups: Dict[Tuple[str, str, str], List[FailureRecord]] = {}
        for record in self._failures:
            error_category = classify_tool_error(record.error_text)
            arg_keys = abstract_arg_keys(record.args)
            key = (record.tool_name, error_category, arg_keys)
            groups.setdefault(key, []).append(record)

        for key, records in groups.items():
            if len(records) >= self.repeat_threshold:
                tool_name, error_category, arg_keys = key
                # Check if arguments are identical (loop) or different (retry)
                # Compare the full error text across records
                identical_args = all(
                    r.error_text == records[0].error_text for r in records
                )
                return FailurePattern(
                    tool_name=tool_name,
                    error_category=error_category,
                    repeat_count=len(records),
                    identical_args=identical_args,
                    records=records,
                )

        return None

    def generate_correction(self, pattern: FailurePattern) -> CorrectionNudge:
        """Generate a correction nudge from a detected pattern."""
        suggestions = self._generate_suggestions(pattern.error_category)
        injection_text = self._format_injection_text(pattern, suggestions)
        return CorrectionNudge(
            tool_name=pattern.tool_name,
            error_category=pattern.error_category,
            repeat_count=pattern.repeat_count,
            identical_args=pattern.identical_args,
            suggestions=suggestions,
            injection_text=injection_text,
            active=True,
        )

    def _generate_suggestions(self, error_category: str) -> List[str]:
        """Generate category-specific suggestions."""
        suggestions_map = {
            "file_not_found": [
                "Verify the file path exists and is accessible",
                "Try listing the directory to find the correct path",
                "Check if the file was created by a previous step",
                "Ensure the file path is not relative to a different working directory",
                "Consider using an absolute path if the relative path is ambiguous",
            ],
            "permission_denied": [
                "Check file permissions and ownership",
                "Try running with elevated privileges if appropriate",
                "Verify the user has read/write access to the target",
                "Check if the file is locked by another process",
                "Ensure the directory is writable",
            ],
            "command_failed": [
                "Check if the command is installed and in PATH",
                "Verify the command syntax is correct",
                "Try running the command manually to see the error",
                "Check if required dependencies are installed",
                "Consider using an alternative command or tool",
            ],
            "network_error": [
                "Check network connectivity",
                "Verify the server is reachable and responding",
                "Try with a different network or proxy configuration",
                "Check if firewall rules are blocking the connection",
                "Consider adding a retry with exponential backoff",
            ],
            "resource_limit": [
                "Check available disk space",
                "Try reducing the size of the operation",
                "Check rate limits and implement backoff",
                "Consider splitting the task into smaller chunks",
                "Check if quotas can be increased",
            ],
            "invalid_input": [
                "Review the expected input format and parameter types",
                "Check that all required parameters are provided",
                "Try with simpler input to isolate the issue",
                "Validate input against the expected schema",
                "Check documentation for the correct parameter format",
            ],
            "tool_not_configured": [
                "Verify the required dependency is installed",
                "Check the tool configuration in config.yaml",
                "Ensure the required API key or token is set",
                "Try installing the missing dependency",
                "Check if the tool requires additional setup steps",
            ],
        }
        return suggestions_map.get(error_category, [
            "Review the error message and try a different approach",
            "Check if the tool is configured correctly",
            "Try with different parameters or a different tool",
        ])

    def _format_injection_text(
        self, pattern: FailurePattern, suggestions: List[str]
    ) -> str:
        """Format the correction nudge for injection."""
        args_note = (
            "Arguments vary across attempts — the model is trying different "
            "inputs but hitting the same error type."
            if not pattern.identical_args
            else "Arguments are identical across attempts — the model is "
            "repeating the same action."
        )
        suggestions_text = "; ".join(suggestions[:3])  # Top 3 suggestions
        return (
            f"[FAILURE CORRECTION: {pattern.tool_name} has failed "
            f"{pattern.repeat_count}x with {pattern.error_category}] "
            f"{args_note} Consider: {suggestions_text}"
        )

    def _prune_old_records(self) -> None:
        """Remove records older than the window."""
        cutoff = self._current_turn_index - self.window + 1
        self._failures = [
            r for r in self._failures if r.turn_index >= cutoff
        ]

    def reset(self) -> None:
        """Reset the tracker (call at the start of each turn)."""
        self._failures.clear()
        self._current_turn_index += 1
        # Prune any stale records that may have been left over
        self._prune_old_records()


# ---------------------------------------------------------------------------
# Module-level tracker (shared across all hook calls)
# ---------------------------------------------------------------------------

_tracker: Optional[FailureTracker] = None


def _get_tracker() -> FailureTracker:
    """Get or create the module-level FailureTracker."""
    global _tracker
    if _tracker is None:
        _tracker = FailureTracker(
            window=_window(),
            repeat_threshold=_repeat_threshold(),
        )
    return _tracker


# ---------------------------------------------------------------------------
# Hook: transform_tool_result
# ---------------------------------------------------------------------------

def _on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    status: str = "",
    error_type: str = "",
    error_message: str = "",
    **_: Any,
) -> Optional[str]:
    """Detect tool errors and inject correction guidance into the result.

    When a tool fails (status="error"), this hook appends a compact
    correction nudge to the result string. The model sees this in the
    next turn and can self-correct.

    Returns None if the tool succeeded or if the plugin is disabled.
    """
    if _plugin_disabled():
        return None

    if status != "error":
        return None

    tracker = _get_tracker()

    # Extract error text from the result
    error_text = extract_error_from_result(result) if result else ""
    if not error_text:
        # Fall back to error_message if provided
        error_text = error_message if error_message else ""

    if not error_text:
        return None

    # Record the failure
    if isinstance(args, dict):
        tracker.record_failure(tool_name, args, error_text)
    else:
        tracker.record_failure(tool_name, {}, error_text)

    # Check for patterns and generate correction
    pattern = tracker.check_for_pattern()
    if pattern is None:
        return None

    correction = tracker.generate_correction(pattern)

    # Append correction to the result
    result_str = str(result) if result else ""
    return result_str + "\n\n" + correction.injection_text


# ---------------------------------------------------------------------------
# Hook: pre_llm_call
# ---------------------------------------------------------------------------

def _on_pre_llm_call(
    session_id: str = "",
    conversation_history: Optional[list] = None,
    **_: Any,
) -> Optional[dict]:
    """Inject failure pattern summary into user message before LLM call.

    Checks if there's a repeating failure pattern and injects a compact
    correction hint into the user message before each LLM call.

    Returns None if no pattern detected or plugin is disabled.
    """
    if _plugin_disabled():
        return None

    tracker = _get_tracker()

    # Check for patterns
    pattern = tracker.check_for_pattern()
    if pattern is None:
        return None

    correction = tracker.generate_correction(pattern)

    # Return context to inject into user message
    return {"context": correction.injection_text}


# ---------------------------------------------------------------------------
# Hook: post_tool_call (observational)
# ---------------------------------------------------------------------------

def _on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    status: str = "",
    error_type: str = "",
    error_message: str = "",
    **_: Any,
) -> None:
    """Observational hook — tracks failures for pattern detection.

    This hook doesn't modify behavior; it just records failures for
    the FailureTracker to detect patterns.
    """
    if _plugin_disabled():
        return

    if status != "error":
        return

    tracker = _get_tracker()

    # Extract error text from the result
    error_text = extract_error_from_result(result) if result else ""
    if not error_text:
        error_text = error_message if error_message else ""

    if not error_text:
        return

    # Record the failure (this is redundant with transform_tool_result,
    # but provides observability for debugging)
    if isinstance(args, dict):
        tracker.record_failure(tool_name, args, error_text)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the self-correction plugin hooks.

    Registers three hooks:
    - transform_tool_result: Detects tool errors and appends correction
      guidance to the result string.
    - pre_llm_call: Detects repeating failure patterns and injects
      correction hints into user messages before each LLM call.
    - post_tool_call: Observational hook that tracks failures for
      pattern detection.
    """
    if _plugin_disabled():
        logger.info("self-correction plugin is disabled via env var")
        return

    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)

    logger.info(
        "self-correction plugin registered (window=%d, threshold=%d)",
        _window(),
        _repeat_threshold(),
    )
