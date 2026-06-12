"""Tool-level failure tracking and self-correction nudges.

When a tool returns an error result (not an API error, but a tool-level
failure like "file not found", "permission denied", "command failed"),
this module tracks the pattern and, when the same tool fails repeatedly
with the same error category, generates a compact correction nudge that
helps the model break out of the failure loop.

Design decisions:

  - **Scope**: tool-level failures only (not API/provider errors — those
    are handled by the error_classifier + retry loop).
  - **Window**: last 4 turns (configurable).  This is a midpoint between
    "current turn only" (too narrow to detect patterns) and "entire
    session" (too much context bloat, risks premature compression).
  - **Correction injection**: primary path appends a compact correction
    block to the last assistant message (preserves role alternation).
    Fallback path injects a synthetic user message (simpler but risks
    breaking alternation if overused).
  - **Pattern detection**: same tool + same error category + same argument
    pattern = loop.  Different arguments on the same tool = productive
    retry, not a loop.

Extracted into its own module so the conversation loop stays readable and
the tracker can be unit-tested in isolation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolFailureRecord:
    """A single tool-level failure event."""

    # Tool name that failed (e.g. "read_file", "terminal").
    tool_name: str = ""
    # The arguments the model passed (abstracted: keys only, not values).
    arg_keys: str = ""
    # The error category (short, normalized).
    error_category: str = ""
    # The raw error message (truncated to 150 chars for compactness).
    error_preview: str = ""
    # The full tool result (stored for correction generation, not sent to model).
    _full_result: str = ""
    # Turn index when this failure occurred (for window management).
    turn_index: int = 0


@dataclass
class FailurePattern:
    """A detected repeating failure pattern."""

    # Tool name that keeps failing.
    tool_name: str = ""
    # How many times this pattern has repeated.
    repeat_count: int = 0
    # The common error category across repetitions.
    error_category: str = ""
    # The common argument pattern (keys) across repetitions.
    arg_pattern: str = ""
    # List of error previews for context.
    error_previews: List[str] = field(default_factory=list)
    # Whether the arguments are identical (same keys AND same values).
    identical_args: bool = False


@dataclass
class CorrectionNudge:
    """A compact correction to inject when a failure pattern is detected."""

    # Whether a correction should be applied.
    active: bool = False
    # The tool that's failing.
    tool_name: str = ""
    # A human-readable description of the failure pattern.
    pattern_description: str = ""
    # Suggested alternative actions (up to 3).
    suggestions: List[str] = field(default_factory=list)
    # The compact text to inject into the next API call.
    injection_text: str = ""
    # Whether to use the synthetic user-message fallback.
    use_fallback: bool = False
    # The fallback user message text (if use_fallback is True).
    fallback_message: str = ""


# ---------------------------------------------------------------------------
# Error category classification
# ---------------------------------------------------------------------------

# Categories for common tool-level failure patterns.
# These are broader than specific error messages — they group similar
# failures so the tracker can detect patterns across different error
# wordings.

_FILE_NOT_FOUND_PATTERNS = [
    "no such file",
    "file not found",
    "file doesn't exist",
    "does not exist",
    "no such directory",
    "directory not found",
    "path not found",
    "enoent",
    "file_missing",
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
    "command failed",
    "exit code",
    "return code",
    "process exited",
    "non-zero exit",
    "command not found",
    "execution failed",
    "failed with",
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
    "too many",
    "limit exceeded",
    "resource exhausted",
]

_INVALID_INPUT_PATTERNS = [
    "invalid argument",
    "invalid parameter",
    "invalid path",
    "invalid format",
    "bad input",
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


def classify_tool_error(error_text: str) -> str:
    """Classify a tool error message into a category.

    Returns a short category string like "file_not_found", "permission_denied",
    "command_failed", etc.  Falls back to "unknown" if no pattern matches.

    Args:
        error_text: The error message from a tool result.

    Returns:
        A category string for pattern matching.
    """
    if not error_text:
        return "empty"

    text_lower = error_text.lower()

    # Check each category's patterns.
    for pattern in _FILE_NOT_FOUND_PATTERNS:
        if pattern in text_lower:
            return "file_not_found"

    for pattern in _PERMISSION_DENIED_PATTERNS:
        if pattern in text_lower:
            return "permission_denied"

    for pattern in _COMMAND_FAILED_PATTERNS:
        if pattern in text_lower:
            return "command_failed"

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


def abstract_arg_keys(function_args: Dict[str, Any]) -> str:
    """Extract sorted argument keys from tool call arguments.

    Returns a string like "path" or "path,command" — abstracted (not
    values) so that calls with different values but the same keys
    produce the same fingerprint.

    Args:
        function_args: The parsed arguments dict from a tool call.

    Returns:
        Comma-separated sorted keys, or empty string if no args.
    """
    if not function_args or not isinstance(function_args, dict):
        return ""
    return ",".join(sorted(function_args.keys()))


def extract_error_from_result(function_result: str) -> str:
    """Extract the error message from a tool result string.

    Handles both structured JSON errors ({"error": "..."}) and plain
    text errors ("Error executing tool 'name': ...").

    Args:
        function_result: The raw tool result string.

    Returns:
        The error message, or empty string if not an error.
    """
    if not function_result:
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
            if data.get("ok") is False and "message" in data:
                return str(data["message"])
            return ""
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain text error: "Error executing tool 'name': ..."
    if isinstance(function_result, str):
        # Check for the canonical error prefix from the tool executor.
        prefix = "Error executing tool '"
        if function_result.startswith(prefix):
            # Extract the tool name and error message.
            rest = function_result[len(prefix):]
            close_quote = rest.find("': ")
            if close_quote != -1:
                return rest[close_quote + 3:]
            # No colon separator — the entire string after the prefix.
            close_single = rest.find("'")
            if close_single != -1:
                return rest[close_single + 1:]
            return rest

        # Check for plain "Error:" prefix.
        if function_result.startswith("Error:"):
            return function_result[len("Error:"):].strip()

        # Check for "error:" prefix (lowercase).
        if function_result.startswith("error:"):
            return function_result[len("error:"):].strip()

    return ""


# ---------------------------------------------------------------------------
# FailureTracker
# ---------------------------------------------------------------------------

class FailureTracker:
    """Tracks tool-level failures and generates correction nudges.

    Configuration (all optional, with sensible defaults):

      window: int — number of recent turns to track (default 4).
      repeat_threshold: int — how many times the same pattern must repeat
          before generating a correction (default 3).
      max_recorded_failures: int — max failure records to keep per turn
          (default 5).

    The tracker is turn-aware: each call to ``record_turn`` advances the
    turn index and prunes records older than the window.
    """

    def __init__(
        self,
        *,
        window: int = 4,
        repeat_threshold: int = 3,
        max_recorded_failures: int = 5,
    ) -> None:
        self.window = max(1, window)
        self.repeat_threshold = max(1, repeat_threshold)
        self.max_recorded_failures = max(1, max_recorded_failures)
        self._failures: List[ToolFailureRecord] = []
        self._current_turn_index: int = 0

    def reset(self) -> None:
        """Reset the tracker (call at the start of each turn)."""
        self._failures.clear()
        self._current_turn_index += 1
        # Prune any stale records that might have survived the clear.
        self._prune_old_records()

    def advance_turn(self) -> None:
        """Advance to the next turn (prunes old records if needed)."""
        self._current_turn_index += 1
        self._prune_old_records()

    def record_failure(
        self,
        tool_name: str,
        function_args: Dict[str, Any],
        function_result: str,
    ) -> None:
        """Record a tool-level failure.

        Args:
            tool_name: The name of the tool that failed.
            function_args: The arguments passed to the tool.
            function_result: The raw tool result string (error message).
        """
        if not tool_name or not function_result:
            return

        error_msg = extract_error_from_result(function_result)
        if not error_msg:
            return

        category = classify_tool_error(error_msg)

        # Truncate error preview for compactness.
        error_preview = error_msg[:150]
        if len(error_msg) > 150:
            error_preview += "..."

        record = ToolFailureRecord(
            tool_name=tool_name,
            arg_keys=abstract_arg_keys(function_args),
            error_category=category,
            error_preview=error_preview,
            _full_result=function_result,
            turn_index=self._current_turn_index,
        )
        self._failures.append(record)

        # Cap per-turn records.
        if len(self._failures) > self.max_recorded_failures:
            self._failures = self._failures[-self.max_recorded_failures:]

    def _prune_old_records(self) -> None:
        """Remove records older than the window."""
        cutoff = self._current_turn_index - self.window + 1
        self._failures = [
            r for r in self._failures if r.turn_index >= cutoff
        ]

    def check_for_pattern(self) -> Optional[FailurePattern]:
        """Check whether a failure pattern has been detected.

        A pattern is detected when the same tool fails with the same
        error category and same argument pattern ``repeat_threshold``
        times within the window.

        Returns a FailurePattern if detected, or None otherwise.
        """
        if len(self._failures) < self.repeat_threshold:
            return None

        # Group failures by (tool_name, error_category, arg_keys).
        groups: Dict[Tuple[str, str, str], List[ToolFailureRecord]] = {}
        for r in self._failures:
            key = (r.tool_name, r.error_category, r.arg_keys)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        # Find the most frequent group.
        best_group = None
        best_count = 0
        for key, records in groups.items():
            if len(records) >= self.repeat_threshold and len(records) > best_count:
                best_count = len(records)
                best_group = (key, records)

        if best_group is None:
            return None

        (tool_name, error_category, arg_keys), records = best_group

        # Check if arguments are identical (same keys AND same values).
        identical = all(
            r._full_result == records[0]._full_result for r in records
        )

        return FailurePattern(
            tool_name=tool_name,
            repeat_count=len(records),
            error_category=error_category,
            arg_pattern=arg_keys,
            error_previews=[r.error_preview for r in records],
            identical_args=identical,
        )

    def generate_correction(self, pattern: FailurePattern) -> CorrectionNudge:
        """Generate a correction nudge for a detected failure pattern.

        Creates a compact correction that helps the model break out of
        the failure loop by suggesting alternative approaches.

        Args:
            pattern: The detected FailurePattern.

        Returns:
            A CorrectionNudge with the correction details.
        """
        suggestions = self._generate_suggestions(
            pattern.tool_name,
            pattern.error_category,
            pattern.arg_pattern,
            pattern.identical_args,
        )

        # Build the injection text (compact, for approach A).
        injection_parts = [
            f"[FAILURE CORRECTION: {pattern.tool_name} has failed "
            f"{pattern.repeat_count}x with {pattern.error_category}]",
        ]
        if pattern.identical_args:
            injection_parts.append(
                "Arguments are identical across all attempts — "
                "retrying with the same input will not help."
            )
        else:
            injection_parts.append(
                "Arguments vary across attempts — the model is trying "
                "different inputs but hitting the same error type."
            )
        injection_parts.append("Consider: " + "; ".join(suggestions[:3]))

        injection_text = " ".join(injection_parts)

        # Build the fallback user message (approach B).
        fallback_message = (
            f"The {pattern.tool_name} tool has failed {pattern.repeat_count} "
            f"times with {pattern.error_category}. The previous attempts used "
            f"arguments: {pattern.arg_pattern}. Please try a different approach "
            f"or strategy instead of repeating the same tool call."
        )

        return CorrectionNudge(
            active=True,
            tool_name=pattern.tool_name,
            pattern_description=(
                f"{pattern.tool_name} failed {pattern.repeat_count}x "
                f"({pattern.error_category}, {pattern.arg_pattern})"
            ),
            suggestions=suggestions,
            injection_text=injection_text,
            use_fallback=False,
            fallback_message=fallback_message,
        )

    def _generate_suggestions(
        self,
        tool_name: str,
        error_category: str,
        arg_pattern: str,
        identical_args: bool,
    ) -> List[str]:
        """Generate context-aware suggestions for breaking a failure pattern.

        Args:
            tool_name: The failing tool name.
            error_category: The error category.
            arg_pattern: The argument key pattern.
            identical_args: Whether all attempts used identical arguments.

        Returns:
            A list of suggestion strings (up to 5).
        """
        suggestions: List[str] = []

        # Generic suggestions based on error category.
        if error_category == "file_not_found":
            suggestions.append("Verify the file path exists and is accessible")
            suggestions.append("Try listing the directory to find the correct path")
            suggestions.append("Check if the file was created by a previous step")

        elif error_category == "permission_denied":
            suggestions.append("Check file/directory permissions")
            suggestions.append("Try running with different privileges")
            suggestions.append("Verify the target path is within allowed directories")

        elif error_category == "command_failed":
            suggestions.append("Check the command syntax and available tools")
            suggestions.append("Try a simpler command to diagnose the issue")
            suggestions.append("Check if the required software is installed")

        elif error_category == "network_error":
            suggestions.append("Check network connectivity")
            suggestions.append("Try a different endpoint or mirror")
            suggestions.append("Verify the URL or host is reachable")

        elif error_category == "resource_limit":
            suggestions.append("Try reducing the scope or size of the request")
            suggestions.append("Check available resources (disk, memory, quota)")
            suggestions.append("Split the task into smaller parts")

        elif error_category == "invalid_input":
            suggestions.append("Review the expected input format and parameter types")
            suggestions.append("Try with simpler or default input values")
            suggestions.append("Check the tool's documentation for valid parameters")

        elif error_category == "tool_not_configured":
            suggestions.append("Check if the required tool/service is installed")
            suggestions.append("Verify the configuration is correct")
            suggestions.append("Try an alternative tool that achieves the same goal")

        # If arguments are identical, suggest varying them.
        if identical_args and arg_pattern:
            suggestions.insert(
                0,
                f"Previous attempts used identical arguments ({arg_pattern}). "
                f"Try different values for these parameters.",
            )

        # If the tool is a file-reading tool, suggest alternatives.
        if tool_name in {"read_file", "search_files"}:
            suggestions.append(
                "Consider using a different tool to achieve the same goal"
            )

        # If the tool is terminal, suggest checking the command.
        if tool_name == "terminal":
            suggestions.append(
                "Try running the command interactively to see the error"
            )

        return suggestions[:5]  # Cap at 5 suggestions.

    def get_failure_summary(self) -> str:
        """Get a compact summary of all tracked failures.

        Returns a short string suitable for injection into the conversation.
        """
        if not self._failures:
            return ""

        # Group by tool name for compactness.
        by_tool: Dict[str, List[ToolFailureRecord]] = {}
        for r in self._failures:
            if r.tool_name not in by_tool:
                by_tool[r.tool_name] = []
            by_tool[r.tool_name].append(r)

        parts = []
        for tool_name, records in sorted(by_tool.items()):
            categories = set(r.error_category for r in records)
            parts.append(
                f"{tool_name}: {len(records)} failure(s) "
                f"({', '.join(sorted(categories))})"
            )

        return " | ".join(parts)

    @property
    def failure_count(self) -> int:
        """Return the number of tracked failures."""
        return len(self._failures)

    @property
    def history(self) -> List[ToolFailureRecord]:
        """Expose the current failure history for debugging."""
        return list(self._failures)


# ---------------------------------------------------------------------------
# Correction injection helpers
# ---------------------------------------------------------------------------

def inject_correction_into_assistant_message(
    messages: List[Dict[str, Any]],
    correction: CorrectionNudge,
) -> bool:
    """Inject a correction nudge into the last assistant message.

    This is the primary (preferred) injection method. It appends the
    correction text to the content of the last assistant message in the
    conversation, preserving role alternation.

    Args:
        messages: The conversation messages list.
        correction: The correction nudge to inject.

    Returns:
        True if injection succeeded, False otherwise.
    """
    if not correction or not correction.active:
        return False

    # Find the last assistant message.
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return False

    # Skip if already injected into this turn.
    if messages[last_assistant_idx].get("_failure_correction_injected"):
        return False

    # Append the correction to the assistant message content.
    msg = messages[last_assistant_idx]
    current_content = msg.get("content")

    if current_content is None:
        msg["content"] = correction.injection_text
    elif isinstance(current_content, str):
        msg["content"] = current_content + "\n\n" + correction.injection_text
    elif isinstance(current_content, list):
        # Multimodal content — append a text block.
        msg["content"] = list(current_content) + [
            {"type": "text", "text": "\n\n" + correction.injection_text}
        ]
    # If content is an unexpected type, silently skip injection.

    # Mark as correction-injected so we don't double-inject.
    msg["_failure_correction_injected"] = True

    return True


def inject_correction_as_user_message(
    messages: List[Dict[str, Any]],
    correction: CorrectionNudge,
) -> bool:
    """Inject a correction as a synthetic user message.

    This is the fallback method. It appends a user message with the
    correction text, which breaks role alternation but is simpler.

    Args:
        messages: The conversation messages list.
        correction: The correction nudge to inject.

    Returns:
        True if injection succeeded, False otherwise.
    """
    if not correction or not correction.active:
        return False

    messages.append({
        "role": "user",
        "content": correction.fallback_message,
        "_failure_correction_synthetic": True,
    })
    return True


def should_use_fallback(messages: List[Dict[str, Any]]) -> bool:
    """Determine whether to use the fallback (user message) injection.

    Returns True if there's no assistant message to inject into, or if
    the last assistant message has already been injected into this turn,
    or if the last message is a user message (to avoid two consecutive
    user messages which breaks role alternation — in this case the caller
    should skip injection entirely).

    Args:
        messages: The conversation messages list.

    Returns:
        True if fallback (or skip) should be used.
    """
    # Check if there's an assistant message to inject into.
    has_assistant = any(
        isinstance(m, dict) and m.get("role") == "assistant"
        for m in messages
    )
    if not has_assistant:
        return True

    # If the last message is a user message, injecting another user
    # message would create two consecutive user messages, breaking
    # role alternation.  Return True to signal the caller should skip
    # injection entirely rather than using the fallback.
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, dict) and last_msg.get("role") == "user":
        return True  # Signal: skip injection (caller handles this).

    # Check if the last assistant message was already injected into.
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            if m.get("_failure_correction_injected"):
                return True
            break

    return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_failure_tracker(
    *,
    window: Optional[int] = None,
    repeat_threshold: Optional[int] = None,
) -> FailureTracker:
    """Create a FailureTracker with defaults from config or sensible fallbacks."""
    from utils import env_int

    if window is None:
        window = env_int("HERMES_FAILURE_TRACKER_WINDOW", 4)
    if repeat_threshold is None:
        repeat_threshold = env_int("HERMES_FAILURE_TRACKER_REPEAT_THRESHOLD", 3)

    return FailureTracker(
        window=window,
        repeat_threshold=repeat_threshold,
    )


__all__ = [
    "FailureTracker",
    "FailurePattern",
    "CorrectionNudge",
    "ToolFailureRecord",
    "create_failure_tracker",
    "inject_correction_into_assistant_message",
    "inject_correction_as_user_message",
    "should_use_fallback",
    "classify_tool_error",
    "abstract_arg_keys",
    "extract_error_from_result",
]
