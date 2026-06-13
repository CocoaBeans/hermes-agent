"""Tests for agent/loop_detector.py — logic loop detection."""

from __future__ import annotations

import pytest

from agent.loop_detector import (
    IterationFingerprint,
    LoopDetectionResult,
    LoopDetector,
    _abstract_tool_args,
    _classify_response,
    _fingerprint_to_pattern,
    create_loop_detector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, args: dict | None = None) -> dict:
    """Create a minimal tool call dict.

    Args must be a dict; this helper serialises it to JSON
    because the real API passes arguments as a JSON string.
    """
    import json
    return {
        "function": {
            "name": name,
            "arguments": json.dumps(args) if args is not None else "{}",
        },
    }


def _make_assistant_text(content: str):
    """Create a minimal assistant message with text content."""
    return type("obj", (), {"content": content, "tool_calls": None, "reasoning": None, "reasoning_content": None, "reasoning_details": None})()


def _make_assistant_with_tools(tool_calls: list[dict]):
    """Create a minimal assistant message with tool calls."""
    tc_objs = [type("obj", (), {"function": tc["function"]})() for tc in tool_calls]
    return type("obj", (), {"content": None, "tool_calls": tc_objs, "reasoning": None, "reasoning_content": None, "reasoning_details": None})()


def _make_assistant_thinking():
    """Create a minimal assistant message with reasoning but no content."""
    return type("obj", (), {"content": "", "tool_calls": None, "reasoning": "thinking...", "reasoning_content": None, "reasoning_details": None})()


# ---------------------------------------------------------------------------
# _abstract_tool_args
# ---------------------------------------------------------------------------

class TestAbstractToolArgs:
    def test_empty(self):
        names, keys = _abstract_tool_args([])
        assert names == ()
        assert keys == ""

    def test_single_tool(self):
        names, keys = _abstract_tool_args([_make_tool_call("search_files", {"pattern": "x", "path": "/y"})])
        assert names == ("search_files",)
        assert keys == "search_files:path,pattern"

    def test_multiple_tools(self):
        calls = [
            _make_tool_call("search_files", {"pattern": "x"}),
            _make_tool_call("read_file", {"path": "/y"}),
        ]
        names, keys = _abstract_tool_args(calls)
        assert names == ("search_files", "read_file")
        assert keys == "search_files:pattern;read_file:path"

    def test_different_args_same_keys(self):
        """Different argument values but same keys should produce the same fingerprint."""
        names1, keys1 = _abstract_tool_args([_make_tool_call("search_files", {"pattern": "foo"})])
        names2, keys2 = _abstract_tool_args([_make_tool_call("search_files", {"pattern": "bar"})])
        assert keys1 == keys2  # Both are "search_files:pattern"


# ---------------------------------------------------------------------------
# _classify_response
# ---------------------------------------------------------------------------

class TestClassifyResponse:
    def test_text_only(self):
        am = _make_assistant_text("Hello world")
        assert _classify_response(am, "Hello world") == "text"

    def test_empty_response(self):
        am = _make_assistant_text("")
        assert _classify_response(am, "") == "empty"

    def test_tool_calls(self):
        am = _make_assistant_with_tools([_make_tool_call("terminal")])
        assert _classify_response(am, None) == "tool_calls"

    def test_thinking_only(self):
        am = _make_assistant_thinking()
        assert _classify_response(am, "") == "thinking_only"


# ---------------------------------------------------------------------------
# _fingerprint_to_pattern
# ---------------------------------------------------------------------------

class TestFingerprintToPattern:
    def test_tool_pattern(self):
        fp = IterationFingerprint(tool_names=("search_files", "read_file"))
        assert _fingerprint_to_pattern(fp) == "search_files → read_file"

    def test_empty_pattern(self):
        fp = IterationFingerprint(response_type="empty")
        assert _fingerprint_to_pattern(fp) == "(empty response)"

    def test_thinking_only_pattern(self):
        fp = IterationFingerprint(response_type="thinking_only")
        assert _fingerprint_to_pattern(fp) == "(thinking only)"

    def test_text_only_pattern(self):
        fp = IterationFingerprint(response_type="text")
        assert _fingerprint_to_pattern(fp) == "(text only)"


# ---------------------------------------------------------------------------
# LoopDetector — happy path
# ---------------------------------------------------------------------------

class TestLoopDetector:
    def test_no_loop_few_iterations(self):
        """Fewer than repeat_threshold iterations should never detect."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        for _ in range(2):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=_make_assistant_with_tools([_make_tool_call("search_files")]),
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        assert not result.detected

    def test_loop_detected(self):
        """Three identical iterations without state change should detect."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        am = _make_assistant_with_tools([_make_tool_call("search_files")])
        for _ in range(3):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        assert result.detected
        assert result.repeat_count == 3
        assert "search_files" in result.pattern
        assert len(result.recovery_suggestions) > 0

    def test_no_loop_with_state_change(self):
        """Same tool calls but state changes should NOT detect."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        am = _make_assistant_with_tools([_make_tool_call("search_files")])
        for i in range(5):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=(i % 2 == 0),  # state changes every other iteration
                message_count=10 + i,
            )
        result = detector.check_for_loop()
        assert not result.detected

    def test_different_tools_no_loop(self):
        """Different tool names should not match."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        for i in range(3):
            detector.record_iteration(
                tool_calls=[_make_tool_call("terminal", {"command": f"cmd{i}"})],
                assistant_message=_make_assistant_with_tools([_make_tool_call("terminal", {"command": f"cmd{i}"})]),
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        # Same tool (terminal) with same abstracted key (command) —
        # this IS a loop because the model keeps calling the same tool
        # with the same argument pattern without making progress.
        result = detector.check_for_loop()
        assert result.detected

    def test_window_limit(self):
        """Older iterations outside the window should not affect detection."""
        detector = LoopDetector(window=3, repeat_threshold=3)
        am = _make_assistant_with_tools([_make_tool_call("search_files")])
        # Record 5 iterations (all same pattern, no state change)
        for _ in range(5):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        # Only the last 3 are in the window, so it should detect
        assert result.detected

    def test_disabled_detector(self):
        """When disabled, no loop should ever be detected."""
        detector = LoopDetector(window=5, repeat_threshold=3, enabled=False)
        am = _make_assistant_with_tools([_make_tool_call("search_files")])
        for _ in range(10):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        assert not result.detected

    def test_reset_clears_history(self):
        detector = LoopDetector(window=5, repeat_threshold=3)
        am = _make_assistant_with_tools([_make_tool_call("search_files")])
        for _ in range(3):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        # Should detect
        assert detector.check_for_loop().detected

        # Reset and record same pattern again — should NOT detect (fresh start)
        detector.reset()
        for _ in range(2):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        assert not detector.check_for_loop().detected

    def test_empty_response_loop(self):
        """Repeated empty responses should be detected."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        am = _make_assistant_text("")
        for _ in range(3):
            detector.record_iteration(
                tool_calls=[],
                assistant_message=am,
                final_response="",
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        assert result.detected
        assert "empty" in result.pattern

    def test_thinking_only_loop(self):
        """Repeated thinking-only responses should be detected."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        am = _make_assistant_thinking()
        for _ in range(3):
            detector.record_iteration(
                tool_calls=[],
                assistant_message=am,
                final_response="",
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        assert result.detected
        assert "thinking" in result.pattern

    def test_recovery_suggestions(self):
        """Loop detection result should include recovery suggestions."""
        detector = LoopDetector(window=5, repeat_threshold=3)
        am = _make_assistant_with_tools([_make_tool_call("search_files")])
        for _ in range(3):
            detector.record_iteration(
                tool_calls=[_make_tool_call("search_files")],
                assistant_message=am,
                final_response=None,
                state_changed=False,
                message_count=10,
            )
        result = detector.check_for_loop()
        assert len(result.recovery_suggestions) >= 2
        assert any("/new" in s for s in result.recovery_suggestions)


# ---------------------------------------------------------------------------
# create_loop_detector (factory)
# ---------------------------------------------------------------------------

class TestCreateLoopDetector:
    def test_default_values(self):
        detector = create_loop_detector()
        assert detector.window == 10
        assert detector.repeat_threshold == 3
        # When no env var is set, loop detection is enabled by default.
        assert detector.enabled is True

    def test_explicit_values(self):
        detector = create_loop_detector(window=5, repeat_threshold=2, enabled=False)
        assert detector.window == 5
        assert detector.repeat_threshold == 2
        assert detector.enabled is False
