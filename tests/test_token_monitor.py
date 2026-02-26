"""Comprehensive tests for the Token Usage Monitor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.token_monitor import TokenMonitor, TokenUsage, _parse_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant_entry(
    input_tokens: int = 1,
    cache_creation: int = 1000,
    cache_read: int = 50000,
    output_tokens: int = 200,
    timestamp: str = "2026-02-26T10:00:00Z",
) -> dict:
    """Build an assistant JSONL entry with usage data."""
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "output_tokens": output_tokens,
                "service_tier": "standard",
            }
        },
    }


def _make_tool_entry() -> dict:
    """Build a tool_use JSONL entry (no usage data)."""
    return {"type": "tool_use", "content": "running command"}


def _make_progress_entry() -> dict:
    """Build a progress JSONL entry (no usage data)."""
    return {"type": "progress", "content": "thinking..."}


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write a list of dicts as JSONL lines to the given path."""
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# TokenUsage dataclass
# ---------------------------------------------------------------------------


class TestTokenUsage:
    """Tests for the TokenUsage frozen dataclass."""

    def test_fields(self) -> None:
        usage = TokenUsage(
            input_tokens=1,
            cache_creation_input_tokens=1205,
            cache_read_input_tokens=97209,
            output_tokens=317,
            total_context=98415,
            timestamp="2026-02-26T10:00:00Z",
        )
        assert usage.input_tokens == 1
        assert usage.cache_creation_input_tokens == 1205
        assert usage.cache_read_input_tokens == 97209
        assert usage.output_tokens == 317
        assert usage.total_context == 98415
        assert usage.timestamp == "2026-02-26T10:00:00Z"

    def test_frozen(self) -> None:
        usage = TokenUsage(
            input_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=0,
            total_context=1,
            timestamp="",
        )
        with pytest.raises(AttributeError):
            usage.input_tokens = 99  # type: ignore[misc]

    def test_total_context_is_independent(self) -> None:
        """total_context is stored, not dynamically computed on the dataclass."""
        usage = TokenUsage(
            input_tokens=10,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
            output_tokens=5,
            total_context=60,  # matches 10+20+30
            timestamp="",
        )
        assert usage.total_context == 60


# ---------------------------------------------------------------------------
# _parse_usage helper
# ---------------------------------------------------------------------------


class TestParseUsage:
    """Tests for the _parse_usage extraction function."""

    def test_valid_assistant_entry(self) -> None:
        entry = _make_assistant_entry(
            input_tokens=1,
            cache_creation=1205,
            cache_read=97209,
            output_tokens=317,
        )
        usage = _parse_usage(entry)
        assert usage is not None
        assert usage.input_tokens == 1
        assert usage.cache_creation_input_tokens == 1205
        assert usage.cache_read_input_tokens == 97209
        assert usage.output_tokens == 317
        assert usage.total_context == 1 + 1205 + 97209

    def test_non_assistant_type(self) -> None:
        entry = {"type": "tool_use", "message": {"usage": {"input_tokens": 5}}}
        assert _parse_usage(entry) is None

    def test_no_type_field(self) -> None:
        entry = {"message": {"usage": {"input_tokens": 5}}}
        assert _parse_usage(entry) is None

    def test_no_message_field(self) -> None:
        entry = {"type": "assistant"}
        assert _parse_usage(entry) is None

    def test_message_not_dict(self) -> None:
        entry = {"type": "assistant", "message": "just a string"}
        assert _parse_usage(entry) is None

    def test_no_usage_field(self) -> None:
        entry = {"type": "assistant", "message": {"content": "hello"}}
        assert _parse_usage(entry) is None

    def test_usage_not_dict(self) -> None:
        entry = {"type": "assistant", "message": {"usage": 42}}
        assert _parse_usage(entry) is None

    def test_missing_optional_fields_default_to_zero(self) -> None:
        """Usage fields that are absent default to 0."""
        entry = {
            "type": "assistant",
            "message": {"usage": {"input_tokens": 100}},
        }
        usage = _parse_usage(entry)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_context == 100

    def test_timestamp_extracted(self) -> None:
        entry = _make_assistant_entry(timestamp="2026-02-26T12:34:56Z")
        usage = _parse_usage(entry)
        assert usage is not None
        assert usage.timestamp == "2026-02-26T12:34:56Z"

    def test_no_timestamp_defaults_to_empty(self) -> None:
        entry = {
            "type": "assistant",
            "message": {"usage": {"input_tokens": 1}},
        }
        usage = _parse_usage(entry)
        assert usage is not None
        assert usage.timestamp == ""

    def test_non_string_timestamp_converted(self) -> None:
        entry = {
            "type": "assistant",
            "timestamp": 1234567890,
            "message": {"usage": {"input_tokens": 1}},
        }
        usage = _parse_usage(entry)
        assert usage is not None
        assert usage.timestamp == "1234567890"

    def test_non_numeric_usage_values_returns_none(self) -> None:
        """Unconvertible usage values -> None (skip entry)."""
        entry = {
            "type": "assistant",
            "message": {"usage": {"input_tokens": "not a number"}},
        }
        assert _parse_usage(entry) is None


# ---------------------------------------------------------------------------
# TokenMonitor — read_latest_usage
# ---------------------------------------------------------------------------


class TestReadLatestUsage:
    """Tests for TokenMonitor.read_latest_usage."""

    def test_single_assistant_entry(self, tmp_path: Path) -> None:
        """Parse a single assistant entry with usage data."""
        path = tmp_path / "session.jsonl"
        entry = _make_assistant_entry(
            input_tokens=1,
            cache_creation=1205,
            cache_read=97209,
            output_tokens=317,
        )
        _write_jsonl(path, [entry])

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)

        assert usage is not None
        assert usage.input_tokens == 1
        assert usage.cache_creation_input_tokens == 1205
        assert usage.cache_read_input_tokens == 97209
        assert usage.output_tokens == 317
        assert usage.total_context == 1 + 1205 + 97209

    def test_latest_of_multiple_assistant_entries(self, tmp_path: Path) -> None:
        """Returns the LAST assistant entry, not the first."""
        path = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(
                input_tokens=1, cache_creation=500, cache_read=10000,
                output_tokens=100, timestamp="2026-02-26T10:00:00Z",
            ),
            _make_tool_entry(),
            _make_assistant_entry(
                input_tokens=1, cache_creation=1200, cache_read=50000,
                output_tokens=200, timestamp="2026-02-26T10:05:00Z",
            ),
            _make_tool_entry(),
            _make_assistant_entry(
                input_tokens=1, cache_creation=1500, cache_read=90000,
                output_tokens=300, timestamp="2026-02-26T10:10:00Z",
            ),
        ]
        _write_jsonl(path, entries)

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)

        assert usage is not None
        assert usage.cache_read_input_tokens == 90000
        assert usage.output_tokens == 300
        assert usage.timestamp == "2026-02-26T10:10:00Z"

    def test_non_assistant_entries_ignored(self, tmp_path: Path) -> None:
        """Only assistant entries with usage are considered."""
        path = tmp_path / "session.jsonl"
        entries = [
            _make_tool_entry(),
            _make_progress_entry(),
            {"type": "system", "content": "init"},
        ]
        _write_jsonl(path, entries)

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is None

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns None."""
        path = tmp_path / "session.jsonl"
        path.write_text("")

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is None

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file returns None without crashing."""
        path = tmp_path / "nonexistent.jsonl"

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is None

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON lines are skipped; valid lines still parsed."""
        path = tmp_path / "session.jsonl"
        valid_entry = _make_assistant_entry(input_tokens=5, cache_creation=100, cache_read=200)
        lines = [
            "this is not json",
            json.dumps(valid_entry),
            "{broken json!!!",
        ]
        path.write_text("\n".join(lines) + "\n")

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)

        assert usage is not None
        assert usage.input_tokens == 5
        assert usage.total_context == 5 + 100 + 200

    def test_assistant_without_usage_skipped(self, tmp_path: Path) -> None:
        """Assistant entry without message.usage is skipped."""
        path = tmp_path / "session.jsonl"
        entries = [
            {"type": "assistant", "message": {"content": "hello"}},
            _make_assistant_entry(input_tokens=10, cache_creation=0, cache_read=0),
        ]
        _write_jsonl(path, entries)

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)

        assert usage is not None
        assert usage.input_tokens == 10

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        """Blank lines in the JSONL file are gracefully skipped."""
        path = tmp_path / "session.jsonl"
        entry = _make_assistant_entry(input_tokens=42)
        content = "\n\n" + json.dumps(entry) + "\n\n"
        path.write_text(content)

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)

        assert usage is not None
        assert usage.input_tokens == 42


# ---------------------------------------------------------------------------
# TokenMonitor — incremental reading (from_position)
# ---------------------------------------------------------------------------


class TestIncrementalReading:
    """Tests for incremental reads via from_position and get_new_position."""

    def test_from_position_zero_reads_entire_file(self, tmp_path: Path) -> None:
        """from_position=0 reads from the beginning."""
        path = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(input_tokens=1, cache_read=10000),
            _make_assistant_entry(input_tokens=2, cache_read=20000),
        ]
        _write_jsonl(path, entries)

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path, from_position=0)

        assert usage is not None
        assert usage.cache_read_input_tokens == 20000  # second/last entry

    def test_from_position_skips_earlier_data(self, tmp_path: Path) -> None:
        """Reading from a later position only sees new entries."""
        path = tmp_path / "session.jsonl"

        # Write first entry
        entry1 = _make_assistant_entry(input_tokens=1, cache_read=10000)
        path.write_text(json.dumps(entry1) + "\n")

        monitor = TokenMonitor()
        usage1 = monitor.read_latest_usage(path, from_position=0)
        assert usage1 is not None
        assert usage1.cache_read_input_tokens == 10000

        pos_after_first = monitor.get_new_position()
        assert pos_after_first > 0

        # Append second entry
        entry2 = _make_assistant_entry(input_tokens=2, cache_read=50000)
        with path.open("a") as f:
            f.write(json.dumps(entry2) + "\n")

        # Read only from where we left off
        usage2 = monitor.read_latest_usage(path, from_position=pos_after_first)
        assert usage2 is not None
        assert usage2.cache_read_input_tokens == 50000

    def test_get_new_position_updates_after_read(self, tmp_path: Path) -> None:
        """get_new_position returns the byte offset after the last read."""
        path = tmp_path / "session.jsonl"
        entry = _make_assistant_entry()
        _write_jsonl(path, [entry])

        monitor = TokenMonitor()
        assert monitor.get_new_position() == 0  # before any read

        monitor.read_latest_usage(path, from_position=0)
        new_pos = monitor.get_new_position()
        assert new_pos > 0
        assert new_pos == path.stat().st_size

    def test_incremental_read_no_new_data(self, tmp_path: Path) -> None:
        """Reading from EOF returns None (no new data)."""
        path = tmp_path / "session.jsonl"
        entry = _make_assistant_entry()
        _write_jsonl(path, [entry])

        monitor = TokenMonitor()
        monitor.read_latest_usage(path, from_position=0)
        pos = monitor.get_new_position()

        # Read again from the same position — no new data
        usage = monitor.read_latest_usage(path, from_position=pos)
        assert usage is None

    def test_position_preserved_on_file_not_found(self, tmp_path: Path) -> None:
        """When file doesn't exist, from_position is preserved."""
        path = tmp_path / "nope.jsonl"
        monitor = TokenMonitor()
        monitor.read_latest_usage(path, from_position=42)
        assert monitor.get_new_position() == 42

    def test_multiple_incremental_reads(self, tmp_path: Path) -> None:
        """Simulate a multi-step incremental read over a growing file."""
        path = tmp_path / "session.jsonl"
        monitor = TokenMonitor()

        # Step 1: write first batch
        entries1 = [
            _make_tool_entry(),
            _make_assistant_entry(input_tokens=1, cache_read=5000),
        ]
        _write_jsonl(path, entries1)

        usage1 = monitor.read_latest_usage(path, from_position=0)
        assert usage1 is not None
        assert usage1.cache_read_input_tokens == 5000
        pos1 = monitor.get_new_position()

        # Step 2: append more entries
        with path.open("a") as f:
            f.write(json.dumps(_make_tool_entry()) + "\n")
            f.write(
                json.dumps(
                    _make_assistant_entry(input_tokens=2, cache_read=30000)
                )
                + "\n"
            )

        usage2 = monitor.read_latest_usage(path, from_position=pos1)
        assert usage2 is not None
        assert usage2.cache_read_input_tokens == 30000
        pos2 = monitor.get_new_position()
        assert pos2 > pos1

        # Step 3: append even more
        with path.open("a") as f:
            f.write(
                json.dumps(
                    _make_assistant_entry(input_tokens=3, cache_read=90000)
                )
                + "\n"
            )

        usage3 = monitor.read_latest_usage(path, from_position=pos2)
        assert usage3 is not None
        assert usage3.cache_read_input_tokens == 90000


# ---------------------------------------------------------------------------
# TokenMonitor — threshold comparison
# ---------------------------------------------------------------------------


class TestThreshold:
    """Tests for is_above_threshold and threshold configuration."""

    def test_above_threshold(self) -> None:
        monitor = TokenMonitor(threshold=50_000)
        usage = TokenUsage(
            input_tokens=1,
            cache_creation_input_tokens=10000,
            cache_read_input_tokens=50000,
            output_tokens=100,
            total_context=60001,
            timestamp="",
        )
        assert monitor.is_above_threshold(usage) is True

    def test_below_threshold(self) -> None:
        monitor = TokenMonitor(threshold=100_000)
        usage = TokenUsage(
            input_tokens=1,
            cache_creation_input_tokens=1000,
            cache_read_input_tokens=5000,
            output_tokens=100,
            total_context=6001,
            timestamp="",
        )
        assert monitor.is_above_threshold(usage) is False

    def test_exactly_at_threshold(self) -> None:
        """Exactly at threshold is NOT above (strict greater-than)."""
        monitor = TokenMonitor(threshold=50_000)
        usage = TokenUsage(
            input_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=0,
            total_context=50_000,
            timestamp="",
        )
        assert monitor.is_above_threshold(usage) is False

    def test_default_threshold(self) -> None:
        monitor = TokenMonitor()
        assert monitor.threshold == 70_000

    def test_custom_threshold(self) -> None:
        monitor = TokenMonitor(threshold=120_000)
        assert monitor.threshold == 120_000

    def test_zero_threshold(self) -> None:
        """Threshold of 0 means any usage is above."""
        monitor = TokenMonitor(threshold=0)
        usage = TokenUsage(
            input_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=0,
            total_context=1,
            timestamp="",
        )
        assert monitor.is_above_threshold(usage) is True


# ---------------------------------------------------------------------------
# Integration with FileTracker
# ---------------------------------------------------------------------------


class TestFileTrackerIntegration:
    """Integration tests using tmp_path to simulate a Claude Code session."""

    def test_full_workflow_with_file_tracker(self, tmp_path: Path) -> None:
        """Simulate the full workflow: FileTracker finds file, TokenMonitor reads it."""
        import uuid

        from src.file_tracker import FileTracker

        # Set up a mock Claude projects directory
        project_dir = tmp_path / "-home-user-myproject"
        project_dir.mkdir()
        session_id = str(uuid.uuid4())
        jsonl_path = project_dir / f"{session_id}.jsonl"

        # Write initial JSONL content
        entries = [
            {"type": "system", "content": "session init"},
            _make_tool_entry(),
            _make_assistant_entry(
                input_tokens=1,
                cache_creation=1205,
                cache_read=97209,
                output_tokens=317,
                timestamp="2026-02-26T10:00:00Z",
            ),
        ]
        _write_jsonl(jsonl_path, entries)

        # FileTracker finds the session
        tracker = FileTracker(claude_dir=tmp_path)
        session = tracker.find_active_session()
        assert session is not None
        assert session.session_id == session_id

        # TokenMonitor reads from the tracked file
        monitor = TokenMonitor(threshold=70_000)
        usage = monitor.read_latest_usage(
            session.file_path,
            from_position=tracker.get_read_position(),
        )

        assert usage is not None
        assert usage.total_context == 1 + 1205 + 97209
        assert usage.timestamp == "2026-02-26T10:00:00Z"
        assert monitor.is_above_threshold(usage) is True

        # Update FileTracker's read position
        tracker.update_read_position(monitor.get_new_position())
        assert tracker.get_read_position() > 0

    def test_incremental_with_file_tracker(self, tmp_path: Path) -> None:
        """FileTracker position feeds TokenMonitor for incremental reads."""
        import uuid

        from src.file_tracker import FileTracker

        project_dir = tmp_path / "-home-user-project"
        project_dir.mkdir()
        session_id = str(uuid.uuid4())
        jsonl_path = project_dir / f"{session_id}.jsonl"

        # First batch
        _write_jsonl(jsonl_path, [
            _make_assistant_entry(input_tokens=1, cache_read=10000),
        ])

        tracker = FileTracker(claude_dir=tmp_path)
        session = tracker.find_active_session()
        assert session is not None

        monitor = TokenMonitor(threshold=50_000)
        usage1 = monitor.read_latest_usage(session.file_path, from_position=0)
        assert usage1 is not None
        assert usage1.cache_read_input_tokens == 10000

        # Update tracker position
        tracker.update_read_position(monitor.get_new_position())

        # Append new data
        with jsonl_path.open("a") as f:
            f.write(json.dumps(
                _make_assistant_entry(input_tokens=2, cache_read=60000)
            ) + "\n")

        # Read incrementally
        usage2 = monitor.read_latest_usage(
            session.file_path,
            from_position=tracker.get_read_position(),
        )
        assert usage2 is not None
        assert usage2.cache_read_input_tokens == 60000
        assert monitor.is_above_threshold(usage2) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_all_zero_tokens(self, tmp_path: Path) -> None:
        """Entry with all zero tokens is valid."""
        path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                }
            },
        }
        _write_jsonl(path, [entry])

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is not None
        assert usage.total_context == 0

    def test_very_large_token_counts(self, tmp_path: Path) -> None:
        """Large token counts are handled correctly."""
        path = tmp_path / "session.jsonl"
        entry = _make_assistant_entry(
            input_tokens=10,
            cache_creation=50000,
            cache_read=150000,
            output_tokens=8000,
        )
        _write_jsonl(path, [entry])

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is not None
        assert usage.total_context == 10 + 50000 + 150000

    def test_mixed_valid_and_invalid_assistant_entries(self, tmp_path: Path) -> None:
        """Monitor picks the latest VALID assistant entry."""
        path = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(input_tokens=1, cache_read=10000),
            {"type": "assistant", "message": "not a dict"},  # invalid
            {"type": "assistant"},  # missing message
        ]
        _write_jsonl(path, entries)

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)

        # Should return the first entry (only valid one)
        assert usage is not None
        assert usage.input_tokens == 1

    def test_only_whitespace_file(self, tmp_path: Path) -> None:
        """File with only whitespace returns None."""
        path = tmp_path / "session.jsonl"
        path.write_text("   \n  \n   \n")

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is None

    def test_extra_fields_in_usage_ignored(self, tmp_path: Path) -> None:
        """Extra/unknown fields in usage don't cause errors."""
        path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 5,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                    "output_tokens": 3,
                    "service_tier": "standard",
                    "some_future_field": 999,
                }
            },
        }
        _write_jsonl(path, [entry])

        monitor = TokenMonitor()
        usage = monitor.read_latest_usage(path)
        assert usage is not None
        assert usage.total_context == 5 + 10 + 20

    def test_concurrent_reads_with_different_monitors(self, tmp_path: Path) -> None:
        """Two monitors reading the same file maintain independent positions."""
        path = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(input_tokens=1, cache_read=10000),
            _make_assistant_entry(input_tokens=2, cache_read=20000),
        ]
        _write_jsonl(path, entries)

        monitor_a = TokenMonitor()
        monitor_b = TokenMonitor()

        # Monitor A reads fully
        usage_a = monitor_a.read_latest_usage(path, from_position=0)
        assert usage_a is not None
        pos_a = monitor_a.get_new_position()

        # Monitor B reads fully (independently)
        usage_b = monitor_b.read_latest_usage(path, from_position=0)
        assert usage_b is not None
        pos_b = monitor_b.get_new_position()

        # Both got the same latest usage and same position
        assert usage_a.cache_read_input_tokens == usage_b.cache_read_input_tokens
        assert pos_a == pos_b

        # Monitor A's position doesn't affect Monitor B
        assert monitor_a.get_new_position() == monitor_b.get_new_position()
