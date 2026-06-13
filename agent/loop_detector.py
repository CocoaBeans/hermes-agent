"""Logic loop detection — recognise when the agent is going in circles.

Extracted into its own module so the conversation loop stays readable and
the detector can be unit-tested in isolation.

A "loop" is detected when the same iteration fingerprint repeats
``repeat_threshold`` times within a rolling window of ``window`` iterations,
without any meaningful state change between repetitions.

An iteration fingerprint captures:
  1. Which tools were called (name + abstracted argument keys).
  2. The response type (text-only, tool calls, empty, thinking-only).
  3. Whether the conversation state changed (new messages, different tool results).

When a loop is detected the conversation loop breaks early and returns a
structured error so the user can recover instead of burning all 90 iterations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IterationFingerprint:
    """A compact, hashable summary of one loop iteration."""

    # Tool names called in this iteration (empty tuple if none).
    tool_names: Tuple[str, ...] = ()
    # Abstracted argument keys (not values) for each tool call.
    # Format: "tool_name:arg1,arg2" separated by ";".
    tool_arg_keys: str = ""
    # Response category.
    response_type: str = "text"
    # Whether the conversation state changed meaningfully (new messages,
    # different tool results, context compression).
    state_changed: bool = False
    # Number of messages in the conversation at the end of this iteration.
    message_count: int = 0
    # Whether context compression ran during this iteration.
    compressed: bool = False


@dataclass
class LoopDetectionResult:
    """Result returned by ``check_for_loop``."""

    detected: bool = False
    pattern: str = ""
    repeat_count: int = 0
    user_message: str = ""
    recovery_suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _abstract_tool_args(tool_calls: List[Dict[str, Any]]) -> Tuple[Tuple[str, ...], str]:
    """Extract tool names and abstracted argument keys from tool calls.

    Returns (tool_names_tuple, arg_keys_string).

    Argument keys are abstracted (not values) so that calls like
    ``search_files(pattern="foo")`` and ``search_files(pattern="bar")``
    produce the same fingerprint — they're the same *pattern* of behaviour.
    """
    if not tool_calls:
        return (), ""

    names: List[str] = []
    arg_parts: List[str] = []

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        if not isinstance(func, dict):
            continue
        name = func.get("name", "?")
        names.append(name)
        try:
            args = json.loads(func.get("arguments", "{}"))
            if isinstance(args, dict):
                # Abstract: sort keys, drop values, keep structure hints.
                keys = sorted(args.keys())
                arg_parts.append(f"{name}:{','.join(keys)}")
            else:
                arg_parts.append(name)
        except (json.JSONDecodeError, TypeError):
            arg_parts.append(name)

    return tuple(names), ";".join(arg_parts)


def _classify_response(
    assistant_message: Any,
    final_response: Optional[str],
) -> str:
    """Classify the type of response from the model.

    Returns one of: "text", "tool_calls", "empty", "thinking_only".
    """
    # Check for tool calls first (they take priority).
    if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
        return "tool_calls"

    # Extract text content.
    content = None
    if final_response is not None:
        content = final_response
    elif hasattr(assistant_message, "content"):
        content = getattr(assistant_message, "content", None)

    if content is None or (isinstance(content, str) and not content.strip()):
        # Check for reasoning/thinking content.
        has_reasoning = (
            getattr(assistant_message, "reasoning", None)
            or getattr(assistant_message, "reasoning_content", None)
            or getattr(assistant_message, "reasoning_details", None)
        )
        if has_reasoning:
            return "thinking_only"
        return "empty"

    text = content if isinstance(content, str) else str(content)
    text = text.strip()

    # Check for thinking blocks.
    has_think = bool(re.search(
        r'<(?:think|thinking|reasoning)[^>]*>', text, re.IGNORECASE
    ))

    if has_think:
        # Check if there's content after the think block.
        after = re.split(
            r'</(?:think|thinking|reasoning)\s*>', text, flags=re.IGNORECASE
        )
        if len(after) > 1 and after[-1].strip():
            return "text"
        return "thinking_only"

    return "text"


def _fingerprint_to_pattern(fp: IterationFingerprint) -> str:
    """Convert a fingerprint to a human-readable pattern string."""
    parts = []
    if fp.tool_names:
        parts.append(" → ".join(fp.tool_names))
    if fp.response_type == "empty":
        parts.append("(empty response)")
    elif fp.response_type == "thinking_only":
        parts.append("(thinking only)")
    elif fp.response_type == "text" and not fp.tool_names:
        parts.append("(text only)")
    return " | ".join(parts) if parts else "(no signal)"


# ---------------------------------------------------------------------------
# LoopDetector
# ---------------------------------------------------------------------------

class LoopDetector:
    """Detects when the agent is stuck in a repeating loop.

    Configuration (all optional, with sensible defaults):

      window: int — rolling window of iterations to track (default 10).
      repeat_threshold: int — how many times the same pattern must repeat
          before flagging a loop (default 3).
      enabled: bool — whether loop detection is active (default True).
    """

    def __init__(
        self,
        *,
        window: int = 10,
        repeat_threshold: int = 3,
        enabled: bool = True,
    ) -> None:
        self.window = max(1, window)
        self.repeat_threshold = max(1, repeat_threshold)
        self.enabled = enabled
        self._history: List[IterationFingerprint] = []

    def reset(self) -> None:
        """Reset the detector (call at the start of each turn)."""
        self._history.clear()

    def record_iteration(
        self,
        tool_calls: List[Dict[str, Any]],
        assistant_message: Any,
        final_response: Optional[str],
        state_changed: bool,
        message_count: int,
        compressed: bool = False,
    ) -> None:
        """Record an iteration's fingerprint for loop detection.

        Args:
            tool_calls: List of tool call dicts from the assistant message.
            assistant_message: The assistant message object (may have
                tool_calls, content, reasoning fields).
            final_response: The text response (if any).
            state_changed: Whether the conversation state changed
                meaningfully (new messages, different results, compression).
            message_count: Total number of messages in the conversation.
            compressed: Whether context compression ran this iteration.
        """
        if not self.enabled:
            return

        tool_names, tool_arg_keys = _abstract_tool_args(tool_calls)
        response_type = _classify_response(assistant_message, final_response)

        fp = IterationFingerprint(
            tool_names=tool_names,
            tool_arg_keys=tool_arg_keys,
            response_type=response_type,
            state_changed=state_changed,
            message_count=message_count,
            compressed=compressed,
        )
        self._history.append(fp)

        # Keep only the last `window` entries.
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

    def check_for_loop(self) -> LoopDetectionResult:
        """Check whether a loop has been detected.

        Returns a LoopDetectionResult with details if a loop is found,
        or a result with detected=False if no loop is detected.
        """
        if not self.enabled or len(self._history) < self.repeat_threshold:
            return LoopDetectionResult()

        # Look for repeating patterns in the tail of the history.
        # A "pattern" is defined by (tool_names, tool_arg_keys, response_type).
        pattern_counts: Dict[str, int] = {}
        pattern_fingerprints: Dict[str, IterationFingerprint] = {}
        patterns_with_state_change: Dict[str, bool] = {}

        for fp in self._history:
            key = f"{fp.tool_names}|{fp.tool_arg_keys}|{fp.response_type}"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
            pattern_fingerprints[key] = fp
            # Track whether this pattern ever had state change.
            if fp.state_changed:
                patterns_with_state_change[key] = True

        # Find the most frequent pattern.
        if not pattern_counts:
            return LoopDetectionResult()

        best_pattern = max(pattern_counts, key=lambda k: pattern_counts[k])
        count = pattern_counts[best_pattern]

        if count >= self.repeat_threshold:
            fp = pattern_fingerprints[best_pattern]
            had_state_change = patterns_with_state_change.get(best_pattern, False)

            # Only flag as a loop if there was NO state change across
            # all repetitions of this pattern.
            if not had_state_change:
                pattern_str = _fingerprint_to_pattern(fp)
                logger.warning(
                    "Loop detected: %s (repeated %dx within window %d)",
                    pattern_str, count, self.window,
                )
                return LoopDetectionResult(
                    detected=True,
                    pattern=pattern_str,
                    repeat_count=count,
                    user_message=(
                        f"Loop detected — the model has repeated the same "
                        f"pattern {count} times without making progress:\n"
                        f"   Pattern: {pattern_str}"
                    ),
                    recovery_suggestions=[
                        "Try /new to start a fresh conversation",
                        "Check if the model has access to the tools it's calling",
                        "Try a different model with /model",
                    ],
                )

        return LoopDetectionResult()

    @property
    def history(self) -> List[IterationFingerprint]:
        """Expose the current history for debugging."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_loop_detector(
    *,
    window: Optional[int] = None,
    repeat_threshold: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> LoopDetector:
    """Create a LoopDetector with defaults from config or sensible fallbacks."""
    from utils import env_int

    # Config precedence: explicit args > env vars > defaults.
    if window is None:
        window = env_int("HERMES_LOOP_DETECTION_WINDOW", 10)
    if repeat_threshold is None:
        repeat_threshold = env_int("HERMES_LOOP_DETECTION_REPEAT_THRESHOLD", 3)
    if enabled is None:
        # env_bool is broken: it always passes "" to is_truthy_value,
        # so when the env var is not set it returns False regardless of
        # the default.  Work around it by checking the env var directly.
        raw = os.getenv("HERMES_LOOP_DETECTION_ENABLED", "")
        enabled = raw.strip().lower() in {"1", "true", "yes", "on"} if raw else True

    return LoopDetector(
        window=window,
        repeat_threshold=repeat_threshold,
        enabled=enabled,
    )


__all__ = [
    "LoopDetector",
    "LoopDetectionResult",
    "IterationFingerprint",
    "create_loop_detector",
]
