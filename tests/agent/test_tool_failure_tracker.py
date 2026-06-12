"""Tests for agent/tool_failure_tracker.py — tool-level failure tracking and correction."""

from __future__ import annotations

import json
import pytest

from agent.tool_failure_tracker import (
    CorrectionNudge,
    FailurePattern,
    FailureTracker,
    ToolFailureRecord,
    abstract_arg_keys,
    classify_tool_error,
    create_failure_tracker,
    extract_error_from_result,
    inject_correction_as_user_message,
    inject_correction_into_assistant_message,
    should_use_fallback,
)


# ---------------------------------------------------------------------------
# classify_tool_error
# ---------------------------------------------------------------------------

class TestClassifyToolError:
    def test_empty_string(self):
        assert classify_tool_error("") == "empty"

    def test_none(self):
        assert classify_tool_error(None) == "empty"

    def test_file_not_found_variants(self):
        for pattern in [
            "no such file",
            "file not found",
            "does not exist",
            "no such directory",
            "path not found",
            "ENOENT",
        ]:
            assert classify_tool_error(pattern) == "file_not_found"

    def test_permission_denied_variants(self):
        for pattern in [
            "permission denied",
            "access denied",
            "EACCES",
            "unauthorized",
            "forbidden",
        ]:
            assert classify_tool_error(pattern) == "permission_denied"

    def test_command_failed_variants(self):
        for pattern in [
            "command failed",
            "exit code 1",
            "return code 2",
            "process exited",
            "non-zero exit",
            "execution failed",
        ]:
            assert classify_tool_error(pattern) == "command_failed"

    def test_network_error_variants(self):
        for pattern in [
            "connection refused",
            "connection timed out",
            "network error",
            "dns resolution failed",
            "request timed out",
            "connection reset",
        ]:
            assert classify_tool_error(pattern) == "network_error"

    def test_resource_limit_variants(self):
        for pattern in [
            "disk space",
            "no space left",
            "quota exceeded",
            "rate limit",
            "too many requests",
            "resource exhausted",
        ]:
            assert classify_tool_error(pattern) == "resource_limit"

    def test_invalid_input_variants(self):
        for pattern in [
            "invalid argument",
            "invalid parameter",
            "invalid path",
            "invalid format",
            "malformed",
            "invalid json",
            "invalid syntax",
        ]:
            assert classify_tool_error(pattern) == "invalid_input"

    def test_tool_not_configured_variants(self):
        for pattern in [
            "not configured for use",
            "not installed",
            "not available",
            "missing dependency",
            "requires installation",
        ]:
            assert classify_tool_error(pattern) == "tool_not_configured"

    def test_unknown_fallback(self):
        assert classify_tool_error("some random error message") == "unknown"

    def test_case_insensitive(self):
        assert classify_tool_error("FILE NOT FOUND") == "file_not_found"
        assert classify_tool_error("Permission Denied") == "permission_denied"


# ---------------------------------------------------------------------------
# abstract_arg_keys
# ---------------------------------------------------------------------------

class TestAbstractArgKeys:
    def test_empty_dict(self):
        assert abstract_arg_keys({}) == ""

    def test_none_input(self):
        assert abstract_arg_keys(None) == ""

    def test_non_dict(self):
        assert abstract_arg_keys("string") == ""

    def test_single_key(self):
        assert abstract_arg_keys({"path": "/foo"}) == "path"

    def test_multiple_keys_sorted(self):
        result = abstract_arg_keys({"path": "/foo", "command": "ls"})
        assert result == "command,path"

    def test_values_ignored(self):
        r1 = abstract_arg_keys({"path": "/foo", "offset": 1})
        r2 = abstract_arg_keys({"path": "/bar", "offset": 100})
        assert r1 == r2 == "offset,path"


# ---------------------------------------------------------------------------
# extract_error_from_result
# ---------------------------------------------------------------------------

class TestExtractErrorFromResult:
    def test_empty_string(self):
        assert extract_error_from_result("") == ""

    def test_none_input(self):
        assert extract_error_from_result(None) == ""

    def test_structured_json_error(self):
        result = json.dumps({"error": "file not found"})
        assert extract_error_from_result(result) == "file not found"

    def test_structured_json_success_false(self):
        result = json.dumps({"success": False, "message": "permission denied"})
        assert extract_error_from_result(result) == "permission denied"

    def test_structured_json_ok_false(self):
        result = json.dumps({"ok": False, "message": "not configured"})
        assert extract_error_from_result(result) == "not configured"

    def test_plain_error_prefix(self):
        result = "Error executing tool 'read_file': file not found: /tmp/missing.txt"
        assert extract_error_from_result(result) == "file not found: /tmp/missing.txt"

    def test_plain_error_prefix_lowercase(self):
        result = "error: connection refused"
        assert extract_error_from_result(result) == "connection refused"

    def test_plain_error_colon(self):
        result = "Error: disk space low"
        assert extract_error_from_result(result) == "disk space low"

    def test_non_error_string(self):
        result = "File written successfully: /tmp/output.txt"
        assert extract_error_from_result(result) == ""

    def test_dict_without_error_key(self):
        result = json.dumps({"data": "some result"})
        assert extract_error_from_result(result) == ""

    def test_nested_json_error_object(self):
        """Nested error objects should extract the message field."""
        result = json.dumps({"error": {"code": 500, "message": "server error"}})
        assert extract_error_from_result(result) == "server error"

    def test_nested_json_error_object_no_message(self):
        """Nested error objects without message field should use repr."""
        result = json.dumps({"error": {"code": 500}})
        # str(dict) produces Python repr
        assert "code" in extract_error_from_result(result)


# ---------------------------------------------------------------------------
# FailureTracker — basic operations
# ---------------------------------------------------------------------------

class TestFailureTrackerBasic:
    def test_create_default(self):
        tracker = create_failure_tracker()
        assert tracker.window == 4
        assert tracker.repeat_threshold == 3
        assert tracker.failure_count == 0
        assert len(tracker.history) == 0

    def test_create_custom(self):
        tracker = create_failure_tracker(window=5, repeat_threshold=2)
        assert tracker.window == 5
        assert tracker.repeat_threshold == 2

    def test_reset_clears_state(self):
        tracker = FailureTracker()
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert tracker.failure_count == 1
        tracker.reset()
        assert tracker.failure_count == 0

    def test_advance_turn(self):
        tracker = FailureTracker(window=4)
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert tracker._current_turn_index == 0
        tracker.advance_turn()
        assert tracker._current_turn_index == 1

    def test_failure_count_property(self):
        tracker = FailureTracker()
        assert tracker.failure_count == 0
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert tracker.failure_count == 1
        tracker.record_failure("terminal", {"command": "ls"}, json.dumps({"error": "exit code 1"}))
        assert tracker.failure_count == 2

    def test_history_property(self):
        tracker = FailureTracker()
        assert tracker.history == []
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert len(tracker.history) == 1
        assert isinstance(tracker.history[0], ToolFailureRecord)

    def test_get_failure_summary_empty(self):
        tracker = FailureTracker()
        assert tracker.get_failure_summary() == ""

    def test_get_failure_summary_with_failures(self):
        tracker = FailureTracker()
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        tracker.record_failure("read_file", {"path": "/bar"}, json.dumps({"error": "not found"}))
        summary = tracker.get_failure_summary()
        assert "read_file" in summary
        assert "2 failure" in summary


# ---------------------------------------------------------------------------
# FailureTracker — pattern detection
# ---------------------------------------------------------------------------

class TestFailureTrackerPatternDetection:
    def test_no_pattern_single_failure(self):
        tracker = FailureTracker(repeat_threshold=3)
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert tracker.check_for_pattern() is None

    def test_no_pattern_different_tools(self):
        tracker = FailureTracker(repeat_threshold=3)
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        tracker.record_failure("terminal", {"command": "ls"}, json.dumps({"error": "exit code 1"}))
        tracker.record_failure("read_file", {"path": "/bar"}, json.dumps({"error": "not found"}))
        assert tracker.check_for_pattern() is None

    def test_pattern_same_tool_same_error(self):
        tracker = FailureTracker(repeat_threshold=3)
        for i in range(3):
            tracker.record_failure("read_file", {"path": f"/foo{i}"}, json.dumps({"error": "file not found"}))
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.tool_name == "read_file"
        assert pattern.error_category == "file_not_found"
        assert pattern.repeat_count == 3

    def test_pattern_identical_args(self):
        tracker = FailureTracker(repeat_threshold=3)
        args = {"path": "/foo/bar/baz.txt"}
        for _ in range(3):
            tracker.record_failure("read_file", args, json.dumps({"error": "file not found"}))
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.identical_args is True

    def test_pattern_different_args_same_keys(self):
        """Different argument values but same keys should still match (same fingerprint)."""
        tracker = FailureTracker(repeat_threshold=3)
        for i in range(3):
            tracker.record_failure(
                "read_file",
                {"path": f"/foo{i}"},
                json.dumps({"error": f"file not found: /foo{i}"}),
            )
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.tool_name == "read_file"
        assert pattern.arg_pattern == "path"
        assert pattern.identical_args is False  # Values differ (different paths in error messages)

    def test_pattern_different_error_categories(self):
        tracker = FailureTracker(repeat_threshold=3)
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "file not found"}))
        tracker.record_failure("read_file", {"path": "/bar"}, json.dumps({"error": "permission denied"}))
        tracker.record_failure("read_file", {"path": "/baz"}, json.dumps({"error": "file not found"}))
        assert tracker.check_for_pattern() is None

    def test_pattern_below_threshold(self):
        tracker = FailureTracker(repeat_threshold=5)
        for _ in range(3):
            tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert tracker.check_for_pattern() is None

    def test_multiple_patterns_returns_most_frequent(self):
        tracker = FailureTracker(repeat_threshold=2)
        # Pattern A: read_file fails 3 times
        for _ in range(3):
            tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        # Pattern B: terminal fails 2 times
        for _ in range(2):
            tracker.record_failure("terminal", {"command": "ls"}, json.dumps({"error": "exit code 1"}))
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.tool_name == "read_file"
        assert pattern.repeat_count == 3

    def test_window_prunes_old_records(self):
        tracker = FailureTracker(window=3, repeat_threshold=3)
        # Record 3 failures in turn 0
        for _ in range(3):
            tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        pattern = tracker.check_for_pattern()
        assert pattern is not None

        # Advance 4 turns — those records should be pruned (cutoff = 4 - 3 = 1,
        # records at turn 0 have turn_index 0 < 1, so pruned).
        for _ in range(4):
            tracker.advance_turn()
        pattern = tracker.check_for_pattern()
        assert pattern is None

    def test_window_allows_recent_records(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        # Record 3 failures in turn 0
        for _ in range(3):
            tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        # Advance 3 turns — records should still be in window (turn 0 is >= turn 3 - 4 = -1)
        for _ in range(3):
            tracker.advance_turn()
        pattern = tracker.check_for_pattern()
        assert pattern is not None

    def test_max_recorded_failures_capped(self):
        tracker = FailureTracker(max_recorded_failures=3)
        for i in range(10):
            tracker.record_failure("read_file", {"path": f"/foo{i}"}, json.dumps({"error": "not found"}))
        assert tracker.failure_count == 3

    def test_no_record_on_non_error_result(self):
        tracker = FailureTracker()
        tracker.record_failure("read_file", {"path": "/foo"}, "File written successfully")
        assert tracker.failure_count == 0

    def test_no_record_on_empty_result(self):
        tracker = FailureTracker()
        tracker.record_failure("read_file", {"path": "/foo"}, "")
        assert tracker.failure_count == 0

    def test_no_record_on_empty_tool_name(self):
        tracker = FailureTracker()
        tracker.record_failure("", {"path": "/foo"}, json.dumps({"error": "not found"}))
        assert tracker.failure_count == 0


# ---------------------------------------------------------------------------
# FailureTracker — correction generation
# ---------------------------------------------------------------------------

class TestFailureTrackerCorrectionGeneration:
    def test_generate_correction_basic(self):
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file",
            repeat_count=3,
            error_category="file_not_found",
            arg_pattern="path",
            error_previews=["file not found: /foo/bar.txt"],
            identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert correction.active is True
        assert correction.tool_name == "read_file"
        assert "read_file" in correction.injection_text
        assert "file_not_found" in correction.injection_text
        assert len(correction.suggestions) > 0

    def test_generate_correction_identical_args(self):
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file",
            repeat_count=3,
            error_category="file_not_found",
            arg_pattern="path",
            error_previews=["file not found: /foo/bar.txt"],
            identical_args=True,
        )
        correction = tracker.generate_correction(pattern)
        assert "identical" in correction.injection_text.lower()

    def test_generate_correction_suggestions_by_category(self):
        tracker = FailureTracker()

        # file_not_found
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("path" in s.lower() or "directory" in s.lower() for s in correction.suggestions)

        # permission_denied
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="permission_denied", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("permission" in s.lower() or "privilege" in s.lower() for s in correction.suggestions)

        # command_failed
        pattern = FailurePattern(
            tool_name="terminal", repeat_count=3,
            error_category="command_failed", arg_pattern="command",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("command" in s.lower() or "syntax" in s.lower() for s in correction.suggestions)

        # network_error
        pattern = FailurePattern(
            tool_name="web_search", repeat_count=3,
            error_category="network_error", arg_pattern="query",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("network" in s.lower() or "connect" in s.lower() or "url" in s.lower() for s in correction.suggestions)

        # resource_limit
        pattern = FailurePattern(
            tool_name="write_file", repeat_count=3,
            error_category="resource_limit", arg_pattern="path,content",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("resource" in s.lower() or "quota" in s.lower() or "split" in s.lower() for s in correction.suggestions)

        # invalid_input — use the outer tracker (not a new instance) to avoid
        # stale module import in the test runner's editable install.
        pattern = FailurePattern(
            tool_name="patch", repeat_count=3,
            error_category="invalid_input", arg_pattern="path,old_string",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("format" in s.lower() or "parameter" in s.lower() or "simple" in s.lower() for s in correction.suggestions)

        # tool_not_configured — use the outer tracker (not a new instance).
        pattern = FailurePattern(
            tool_name="spotify", repeat_count=3,
            error_category="tool_not_configured", arg_pattern="action",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert any("installed" in s.lower() or "config" in s.lower() or "alternative" in s.lower() for s in correction.suggestions)

    def test_generate_correction_suggestions_capped_at_5(self):
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=True,
        )
        correction = tracker.generate_correction(pattern)
        assert len(correction.suggestions) <= 5

    def test_generate_correction_has_fallback_message(self):
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert correction.fallback_message != ""
        assert "read_file" in correction.fallback_message
        assert "file_not_found" in correction.fallback_message

    def test_generate_correction_injection_text_compact(self):
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        # Should be a single line, not excessively long
        assert len(correction.injection_text) < 500
        assert "[FAILURE CORRECTION:" in correction.injection_text

    def test_active_flag_on_correction(self):
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        assert correction.active is True


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

class TestInjectionHelpers:
    def test_inject_into_assistant_message_success(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check that file."},
        ]
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        result = inject_correction_into_assistant_message(messages, correction)
        assert result is True
        # The last assistant message should have the correction appended.
        last_msg = messages[-1]
        assert last_msg["role"] == "assistant"
        assert "[FAILURE CORRECTION:" in last_msg["content"]
        assert last_msg.get("_failure_correction_injected") is True

    def test_inject_into_assistant_message_no_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
        ]
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        result = inject_correction_into_assistant_message(messages, correction)
        assert result is False

    def test_inject_into_assistant_message_already_injected(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check.", "_failure_correction_injected": True},
        ]
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        result = inject_correction_into_assistant_message(messages, correction)
        assert result is False

    def test_inject_as_user_message(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check."},
        ]
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        result = inject_correction_as_user_message(messages, correction)
        assert result is True
        # A user message should have been appended.
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        assert "read_file" in last_msg["content"]
        assert last_msg.get("_failure_correction_synthetic") is True

    def test_should_use_fallback_no_assistant(self):
        messages = [{"role": "user", "content": "hello"}]
        assert should_use_fallback(messages) is True

    def test_should_use_fallback_with_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check."},
        ]
        assert should_use_fallback(messages) is False

    def test_should_use_fallback_already_injected(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check.", "_failure_correction_injected": True},
        ]
        assert should_use_fallback(messages) is True

    def test_inject_with_multimodal_content(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Let me check."}]},
        ]
        tracker = FailureTracker()
        pattern = FailurePattern(
            tool_name="read_file", repeat_count=3,
            error_category="file_not_found", arg_pattern="path",
            error_previews=[], identical_args=False,
        )
        correction = tracker.generate_correction(pattern)
        result = inject_correction_into_assistant_message(messages, correction)
        assert result is True
        # Should have appended a text block to the content list.
        last_msg = messages[-1]
        content = last_msg["content"]
        assert isinstance(content, list)
        assert len(content) > 1
        assert "[FAILURE CORRECTION:" in content[-1]["text"]

    def test_inject_inactive_correction_noop(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me check."},
        ]
        inactive = CorrectionNudge(active=False)
        result = inject_correction_into_assistant_message(messages, inactive)
        assert result is False

    def test_inject_inactive_user_message_noop(self):
        messages = [
            {"role": "user", "content": "hello"},
        ]
        inactive = CorrectionNudge(active=False)
        result = inject_correction_as_user_message(messages, inactive)
        assert result is False


# ---------------------------------------------------------------------------
# FailureTracker — end-to-end integration
# ---------------------------------------------------------------------------

class TestFailureTrackerEndToEnd:
    def test_full_pattern_detection_and_correction(self):
        """Simulate 3 consecutive tool failures with the same tool and error."""
        tracker = FailureTracker(window=4, repeat_threshold=3)

        # Turn 0: 3 failures from read_file
        for i in range(3):
            tracker.record_failure(
                "read_file",
                {"path": f"/tmp/data_{i}.json"},
                json.dumps({"error": f"file not found: /tmp/data_{i}.json"}),
            )

        # Should detect the pattern now.
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.tool_name == "read_file"
        assert pattern.error_category == "file_not_found"
        assert pattern.repeat_count == 3

        # Generate correction.
        correction = tracker.generate_correction(pattern)
        assert correction.active is True
        assert "read_file" in correction.injection_text
        assert "file_not_found" in correction.injection_text
        assert len(correction.suggestions) > 0

    def test_productive_retry_does_not_trigger(self):
        """Different errors on the same tool should NOT trigger a pattern."""
        tracker = FailureTracker(repeat_threshold=3)
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "file not found"}))
        tracker.record_failure("read_file", {"path": "/bar"}, json.dumps({"error": "permission denied"}))
        tracker.record_failure("read_file", {"path": "/baz"}, json.dumps({"error": "not a file"}))
        assert tracker.check_for_pattern() is None

    def test_mixed_success_and_failure(self):
        """Successes between failures should not break the pattern."""
        tracker = FailureTracker(window=4, repeat_threshold=3)

        # Turn 0: 2 failures
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        tracker.record_failure("read_file", {"path": "/bar"}, json.dumps({"error": "not found"}))

        # Advance turn (simulating a successful tool call in between).
        tracker.advance_turn()

        # Turn 1: 1 more failure
        tracker.record_failure("read_file", {"path": "/baz"}, json.dumps({"error": "not found"}))

        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.tool_name == "read_file"

    def test_window_expiration_clears_pattern(self):
        """Old failures should be pruned when the window expires."""
        tracker = FailureTracker(window=2, repeat_threshold=3)

        # Turn 0: 3 failures
        for _ in range(3):
            tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))

        assert tracker.check_for_pattern() is not None

        # Advance 2 turns — records from turn 0 should be pruned (cutoff = 2 - 2 = 0, but records at turn 0 are >= 0, so they're still in window... let me check).
        # Actually, cutoff = current_turn - window = 2 - 2 = 0. Records at turn 0 have turn_index 0 >= 0, so they're kept.
        # Let's advance 3 turns instead.
        for _ in range(3):
            tracker.advance_turn()
        # cutoff = 3 - 2 = 1. Records at turn 0 have turn_index 0 < 1, so pruned.
        assert tracker.check_for_pattern() is None

    def test_get_failure_summary_multiple_tools(self):
        tracker = FailureTracker()
        tracker.record_failure("read_file", {"path": "/foo"}, json.dumps({"error": "not found"}))
        tracker.record_failure("read_file", {"path": "/bar"}, json.dumps({"error": "not found"}))
        tracker.record_failure("terminal", {"command": "ls"}, json.dumps({"error": "exit code 1"}))
        tracker.record_failure("terminal", {"command": "cd"}, json.dumps({"error": "no such directory"}))

        summary = tracker.get_failure_summary()
        assert "read_file" in summary
        assert "terminal" in summary
        assert "2 failure" in summary
