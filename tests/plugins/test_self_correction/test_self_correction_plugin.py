"""Tests for the self-correction plugin.

Tests cover:
- Error classification (all categories)
- Argument abstraction
- Error extraction from various result formats
- FailureTracker operations (record, check, prune, reset)
- Pattern detection (same tool, same error, same args)
- Correction generation (category-specific suggestions)
- Plugin hooks (transform_tool_result, pre_llm_call, post_tool_call)
- Configuration (window, threshold, disable)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add the plugins directory to the path
_plugins_dir = str(Path(__file__).parent.parent.parent.parent / "plugins")
if _plugins_dir not in sys.path:
    sys.path.insert(0, _plugins_dir)

# Import the plugin module
from self_correction import (
    FailureTracker,
    FailurePattern,
    CorrectionNudge,
    _on_transform_tool_result,
    _on_pre_llm_call,
    _on_post_tool_call,
    _window,
    _repeat_threshold,
    _plugin_disabled,
    classify_tool_error,
    abstract_arg_keys,
    extract_error_from_result,
)


class TestErrorClassification:
    """Test classify_tool_error function."""

    def test_file_not_found_variants(self):
        for pattern in [
            "no such file",
            "file not found",
            "file doesn't exist",
            "does not exist",
            "no such directory",
            "directory not found",
            "path not found",
            "enoent",
        ]:
            assert classify_tool_error(pattern) == "file_not_found"

    def test_permission_denied_variants(self):
        for pattern in [
            "permission denied",
            "access denied",
            "eacces",
            "unauthorized",
            "forbidden",
            "not allowed",
        ]:
            assert classify_tool_error(pattern) == "permission_denied"

    def test_command_failed_variants(self):
        for pattern in [
            "command not found",
            "command failed",
            "command timed out",
            "exit code",
            "return code",
            "non-zero exit",
        ]:
            assert classify_tool_error(pattern) == "command_failed"

    def test_network_error_variants(self):
        for pattern in [
            "connection refused",
            "connection timed out",
            "network error",
            "network unreachable",
            "dns resolution failed",
            "host not found",
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
            "dependency not found",
            "requires installation",
            "setup required",
        ]:
            assert classify_tool_error(pattern) == "tool_not_configured"

    def test_unknown_category(self):
        assert classify_tool_error("some random error") == "unknown"

    def test_case_insensitive(self):
        assert classify_tool_error("FILE NOT FOUND") == "file_not_found"
        assert classify_tool_error("Permission Denied") == "permission_denied"


class TestArgumentAbstraction:
    """Test abstract_arg_keys function."""

    def test_empty_dict(self):
        assert abstract_arg_keys({}) == ""

    def test_single_key(self):
        assert abstract_arg_keys({"path": "/tmp/test.txt"}) == "path"

    def test_multiple_keys(self):
        assert abstract_arg_keys({"path": "/tmp", "content": "data"}) == "content,path"

    def test_non_dict(self):
        assert abstract_arg_keys("string") == ""
        assert abstract_arg_keys(123) == ""
        assert abstract_arg_keys(None) == ""


class TestErrorExtraction:
    """Test extract_error_from_result function."""

    def test_plain_error_string(self):
        result = "Error: file not found: /tmp/missing.txt"
        assert extract_error_from_result(result) == "file not found: /tmp/missing.txt"

    def test_structured_json_error(self):
        result = json.dumps({"error": "file not found"})
        assert extract_error_from_result(result) == "file not found"

    def test_structured_json_success_false(self):
        result = json.dumps({"success": False, "message": "connection refused"})
        assert extract_error_from_result(result) == "connection refused"

    def test_structured_json_ok_false(self):
        result = json.dumps({"ok": False, "message": "rate limit exceeded"})
        assert extract_error_from_result(result) == "rate limit exceeded"

    def test_nested_json_error_object(self):
        result = json.dumps({"error": {"code": 500, "message": "internal server error"}})
        assert extract_error_from_result(result) == "internal server error"

    def test_dict_without_error_key(self):
        result = json.dumps({"data": "some result"})
        assert extract_error_from_result(result) == ""

    def test_non_string_input(self):
        assert extract_error_from_result(None) == ""
        assert extract_error_from_result(123) == ""
        assert extract_error_from_result({}) == ""

    def test_no_error_prefix(self):
        result = "Operation completed successfully"
        assert extract_error_from_result(result) == ""


class TestFailureTracker:
    """Test FailureTracker class."""

    def test_initial_state(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        assert tracker.failure_count == 0
        assert tracker.history == []

    def test_record_failure(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        tracker.record_failure("write_file", {"path": "/tmp/test.txt"}, "file not found")
        assert tracker.failure_count == 1
        assert len(tracker.history) == 1
        assert tracker.history[0].tool_name == "write_file"
        assert tracker.history[0].error_text == "file not found"

    def test_prune_old_records(self):
        tracker = FailureTracker(window=2, repeat_threshold=3)
        # Add 3 failures
        for i in range(3):
            tracker.record_failure("write_file", {"path": f"/tmp/test{i}.txt"}, "file not found")
        # After reset, turn_index increments and old records are pruned
        tracker.reset()
        assert tracker.failure_count == 0

    def test_reset(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        tracker.record_failure("write_file", {"path": "/tmp/test.txt"}, "file not found")
        assert tracker.failure_count == 1
        tracker.reset()
        assert tracker.failure_count == 0

    def test_check_for_pattern_no_pattern(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        tracker.record_failure("write_file", {"path": "/tmp/test1.txt"}, "file not found")
        tracker.record_failure("write_file", {"path": "/tmp/test2.txt"}, "file not found")
        assert tracker.check_for_pattern() is None

    def test_check_for_pattern_detected(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        for i in range(3):
            tracker.record_failure("write_file", {"path": f"/tmp/test{i}.txt"}, "file not found")
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.tool_name == "write_file"
        assert pattern.error_category == "file_not_found"
        assert pattern.repeat_count == 3

    def test_check_for_pattern_identical_args(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        for i in range(3):
            tracker.record_failure("write_file", {"path": "/tmp/test.txt"}, "file not found")
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        assert pattern.identical_args is True

    def test_check_for_pattern_different_args(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        # Use different error messages to simulate different args
        for i in range(3):
            tracker.record_failure(
                "write_file",
                {"path": f"/tmp/test{i}.txt"},
                f"file not found: /tmp/test{i}.txt",
            )
        pattern = tracker.check_for_pattern()
        assert pattern is not None
        # Different error text = different args = not identical_args
        assert pattern.identical_args is False

    def test_window_pruning(self):
        tracker = FailureTracker(window=2, repeat_threshold=3)
        # Add 3 failures
        for i in range(3):
            tracker.record_failure("write_file", {"path": f"/tmp/test{i}.txt"}, "file not found")
        # After reset, turn_index increments and old records are pruned
        tracker.reset()
        assert tracker.failure_count == 0


class TestCorrectionGeneration:
    """Test correction nudge generation."""

    def test_generate_correction_file_not_found(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        pattern = FailurePattern(
            tool_name="write_file",
            error_category="file_not_found",
            repeat_count=3,
            identical_args=False,
            records=[],
        )
        correction = tracker.generate_correction(pattern)
        assert correction.tool_name == "write_file"
        assert correction.error_category == "file_not_found"
        assert correction.repeat_count == 3
        assert correction.identical_args is False
        assert len(correction.suggestions) > 0
        assert "file" in correction.injection_text.lower()

    def test_generate_correction_permission_denied(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        pattern = FailurePattern(
            tool_name="terminal",
            error_category="permission_denied",
            repeat_count=3,
            identical_args=True,
            records=[],
        )
        correction = tracker.generate_correction(pattern)
        assert correction.error_category == "permission_denied"
        assert "permission" in correction.injection_text.lower()

    def test_generate_correction_network_error(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        pattern = FailurePattern(
            tool_name="web_search",
            error_category="network_error",
            repeat_count=3,
            identical_args=False,
            records=[],
        )
        correction = tracker.generate_correction(pattern)
        assert correction.error_category == "network_error"
        assert "network" in correction.injection_text.lower()

    def test_injection_text_format(self):
        tracker = FailureTracker(window=4, repeat_threshold=3)
        pattern = FailurePattern(
            tool_name="write_file",
            error_category="file_not_found",
            repeat_count=3,
            identical_args=False,
            records=[],
        )
        correction = tracker.generate_correction(pattern)
        assert "[FAILURE CORRECTION:" in correction.injection_text
        assert "write_file" in correction.injection_text
        assert "file_not_found" in correction.injection_text


class TestPluginHooks:
    """Test plugin hooks."""

    def test_transform_tool_result_success(self):
        """Test that transform_tool_result returns None for successful tools."""
        result = _on_transform_tool_result(
            tool_name="write_file",
            args={"path": "/tmp/test.txt"},
            result="File written successfully",
            status="completed",
        )
        assert result is None

    def test_transform_tool_result_error(self):
        """Test that transform_tool_result appends correction for errors."""
        # Reset the module-level tracker
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            # Record 3 failures
            for i in range(3):
                _on_transform_tool_result(
                    tool_name="write_file",
                    args={"path": f"/tmp/test{i}.txt"},
                    result=json.dumps({"error": "file not found"}),
                    status="error",
                )

            # The 3rd call should return a correction
            result = _on_transform_tool_result(
                tool_name="write_file",
                args={"path": "/tmp/test3.txt"},
                result=json.dumps({"error": "file not found"}),
                status="error",
            )
            assert result is not None
            assert "[FAILURE CORRECTION:" in result
        finally:
            self_correction._tracker = old_tracker

    def test_pre_llm_call_no_pattern(self):
        """Test that pre_llm_call returns None when no pattern detected."""
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            result = _on_pre_llm_call(session_id="test")
            assert result is None
        finally:
            self_correction._tracker = old_tracker

    def test_pre_llm_call_with_pattern(self):
        """Test that pre_llm_call returns context when pattern detected."""
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            # Record 3 failures
            for i in range(3):
                _on_transform_tool_result(
                    tool_name="write_file",
                    args={"path": f"/tmp/test{i}.txt"},
                    result=json.dumps({"error": "file not found"}),
                    status="error",
                )

            # The 3rd call should return context
            result = _on_pre_llm_call(session_id="test")
            assert result is not None
            assert "context" in result
            assert "[FAILURE CORRECTION:" in result["context"]
        finally:
            self_correction._tracker = old_tracker

    def test_post_tool_call_error(self):
        """Test that post_tool_call records errors but returns None."""
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            result = _on_post_tool_call(
                tool_name="write_file",
                args={"path": "/tmp/test.txt"},
                result=json.dumps({"error": "file not found"}),
                status="error",
            )
            assert result is None
            assert self_correction._tracker.failure_count == 1
        finally:
            self_correction._tracker = old_tracker


class TestPluginConfiguration:
    """Test plugin configuration."""

    def test_default_window(self):
        # Ensure env var is not set
        old_val = os.environ.pop("HERMES_SELF_CORRECTION_WINDOW", None)
        try:
            assert _window() == 4
        finally:
            if old_val is not None:
                os.environ["HERMES_SELF_CORRECTION_WINDOW"] = old_val

    def test_custom_window(self):
        os.environ["HERMES_SELF_CORRECTION_WINDOW"] = "10"
        try:
            assert _window() == 10
        finally:
            os.environ.pop("HERMES_SELF_CORRECTION_WINDOW", None)

    def test_default_threshold(self):
        # Ensure env var is not set
        old_val = os.environ.pop("HERMES_SELF_CORRECTION_REPEAT_THRESHOLD", None)
        try:
            assert _repeat_threshold() == 3
        finally:
            if old_val is not None:
                os.environ["HERMES_SELF_CORRECTION_REPEAT_THRESHOLD"] = old_val

    def test_custom_threshold(self):
        os.environ["HERMES_SELF_CORRECTION_REPEAT_THRESHOLD"] = "5"
        try:
            assert _repeat_threshold() == 5
        finally:
            os.environ.pop("HERMES_SELF_CORRECTION_REPEAT_THRESHOLD", None)

    def test_plugin_disabled(self):
        os.environ["HERMES_SELF_CORRECTION_DISABLE"] = "1"
        try:
            assert _plugin_disabled() is True
        finally:
            os.environ.pop("HERMES_SELF_CORRECTION_DISABLE", None)

    def test_plugin_not_disabled(self):
        os.environ.pop("HERMES_SELF_CORRECTION_DISABLE", None)
        assert _plugin_disabled() is False


class TestIntegration:
    """Integration tests for the full plugin workflow."""

    def test_full_workflow(self):
        """Test the full workflow: record failures -> detect pattern -> generate correction."""
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            # Record 3 failures
            for i in range(3):
                _on_transform_tool_result(
                    tool_name="write_file",
                    args={"path": f"/tmp/test{i}.txt"},
                    result=json.dumps({"error": "file not found"}),
                    status="error",
                )

            # Check for pattern
            pattern = self_correction._tracker.check_for_pattern()
            assert pattern is not None
            assert pattern.tool_name == "write_file"
            assert pattern.error_category == "file_not_found"
            assert pattern.repeat_count == 3

            # Generate correction
            correction = self_correction._tracker.generate_correction(pattern)
            assert correction.active is True
            assert "write_file" in correction.injection_text
            assert "file_not_found" in correction.injection_text
            assert len(correction.suggestions) > 0

            # Verify injection text format
            assert "[FAILURE CORRECTION:" in correction.injection_text
            assert "Arguments are identical across attempts" in correction.injection_text
        finally:
            self_correction._tracker = old_tracker

    def test_productive_retry_not_flagged(self):
        """Test that productive retries (different args) are not flagged as loops."""
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            # Record 3 failures with different paths (productive retry)
            for i in range(3):
                _on_transform_tool_result(
                    tool_name="write_file",
                    args={"path": f"/tmp/test{i}.txt"},
                    result=json.dumps({"error": f"file not found: /tmp/test{i}.txt"}),
                    status="error",
                )

            pattern = self_correction._tracker.check_for_pattern()
            assert pattern is not None
            assert pattern.identical_args is False  # Different paths = productive retry

            correction = self_correction._tracker.generate_correction(pattern)
            assert "Arguments vary across attempts" in correction.injection_text
        finally:
            self_correction._tracker = old_tracker

    def test_identical_args_flagged_as_loop(self):
        """Test that identical retries are flagged as loops."""
        import self_correction
        old_tracker = self_correction._tracker
        self_correction._tracker = FailureTracker(window=4, repeat_threshold=3)

        try:
            # Record 3 failures with identical args (loop)
            for i in range(3):
                _on_transform_tool_result(
                    tool_name="write_file",
                    args={"path": "/tmp/test.txt"},
                    result=json.dumps({"error": "file not found"}),
                    status="error",
                )

            pattern = self_correction._tracker.check_for_pattern()
            assert pattern is not None
            assert pattern.identical_args is True  # Same args = loop

            correction = self_correction._tracker.generate_correction(pattern)
            assert "Arguments are identical across attempts" in correction.injection_text
        finally:
            self_correction._tracker = old_tracker
