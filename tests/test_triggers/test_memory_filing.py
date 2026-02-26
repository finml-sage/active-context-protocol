"""Comprehensive tests for the Memory Filing Reminder Trigger."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.triggers.compaction import TriggerDecision
from src.triggers.memory_filing import MemoryFilingTrigger, MilestoneEvent


# ---------------------------------------------------------------------------
# Helpers — JSONL entry builders
# ---------------------------------------------------------------------------


def _tool_result_entry(
    content: str, timestamp: str = "2026-02-26T12:00:00Z"
) -> str:
    """Build a JSONL line for a user entry with a tool_result."""
    entry = {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": [
                {
                    "tool_use_id": "toolu_test",
                    "type": "tool_result",
                    "content": content,
                }
            ],
        },
    }
    return json.dumps(entry)


def _tool_call_entry(
    name: str, command: str, timestamp: str = "2026-02-26T12:00:00Z"
) -> str:
    """Build a JSONL line for an assistant entry with a tool_use."""
    entry = {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_test",
                    "name": name,
                    "input": {"command": command},
                }
            ],
        },
    }
    return json.dumps(entry)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    """Write JSONL lines to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# MilestoneEvent dataclass
# ---------------------------------------------------------------------------


class TestMilestoneEvent:
    """Tests for the MilestoneEvent dataclass."""

    def test_fields(self) -> None:
        event = MilestoneEvent(
            event_type="pr_merge",
            timestamp="2026-02-26T12:00:00Z",
            details="Merged PR #42",
        )
        assert event.event_type == "pr_merge"
        assert event.timestamp == "2026-02-26T12:00:00Z"
        assert event.details == "Merged PR #42"

    def test_frozen(self) -> None:
        event = MilestoneEvent(
            event_type="commit",
            timestamp="",
            details="test",
        )
        with pytest.raises(AttributeError):
            event.event_type = "pr_merge"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Milestone detection — PR merge
# ---------------------------------------------------------------------------


class TestDetectPRMerge:
    """Tests for detecting PR merge events."""

    def test_merged_in_tool_result(self, tmp_path: Path) -> None:
        """Detects 'Merged' in tool result content."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Merged pull request #42 into main"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert len(events) >= 1
        pr_events = [e for e in events if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1
        assert "Merged" in pr_events[0].details

    def test_gh_pr_merge_in_tool_call(self, tmp_path: Path) -> None:
        """Detects gh pr merge in tool call input."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_call_entry("Bash", "gh pr merge 42 --squash"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1

    def test_gh_pr_merge_in_tool_result(self, tmp_path: Path) -> None:
        """Detects gh pr merge output in tool result."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Running: gh pr merge 42\nMerged."),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1


# ---------------------------------------------------------------------------
# Milestone detection — issue close
# ---------------------------------------------------------------------------


class TestDetectIssueClose:
    """Tests for detecting issue close events."""

    def test_gh_issue_close_in_tool_result(self, tmp_path: Path) -> None:
        """Detects gh issue close in tool result."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("gh issue close 15 completed"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        issue_events = [e for e in events if e.event_type == "issue_close"]
        assert len(issue_events) >= 1

    def test_gh_issue_close_in_tool_call(self, tmp_path: Path) -> None:
        """Detects gh issue close in tool call input."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_call_entry("Bash", "gh issue close 15"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        issue_events = [e for e in events if e.event_type == "issue_close"]
        assert len(issue_events) >= 1

    def test_closed_issue_in_result(self, tmp_path: Path) -> None:
        """Detects 'Closed issue' in tool result content."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Closed issue #15 as completed"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        issue_events = [e for e in events if e.event_type == "issue_close"]
        assert len(issue_events) >= 1


# ---------------------------------------------------------------------------
# Milestone detection — commit
# ---------------------------------------------------------------------------


class TestDetectCommit:
    """Tests for detecting commit events."""

    def test_git_commit_output_in_result(self, tmp_path: Path) -> None:
        """Detects git commit output format '[branch hash] message'."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry(
                "[main abc1234] Add memory filing trigger\n"
                " 2 files changed, 150 insertions(+)"
            ),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        commit_events = [e for e in events if e.event_type == "commit"]
        assert len(commit_events) >= 1

    def test_git_commit_in_tool_call(self, tmp_path: Path) -> None:
        """Detects git commit in tool call input."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_call_entry("Bash", "git commit -m 'Fix bug in parser'"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        commit_events = [e for e in events if e.event_type == "commit"]
        assert len(commit_events) >= 1

    def test_create_mode_in_result(self, tmp_path: Path) -> None:
        """Detects 'create mode' in tool result (appears in commit output)."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry(
                "[feat/new a1b2c3d] Add new feature\n"
                " create mode 100644 src/new_module.py"
            ),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        commit_events = [e for e in events if e.event_type == "commit"]
        assert len(commit_events) >= 1


# ---------------------------------------------------------------------------
# Milestone detection — PR create
# ---------------------------------------------------------------------------


class TestDetectPRCreate:
    """Tests for detecting PR create events."""

    def test_gh_pr_create_in_tool_call(self, tmp_path: Path) -> None:
        """Detects gh pr create in tool call input."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_call_entry("Bash", "gh pr create --title 'Fix' --body 'desc'"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_create"]
        assert len(pr_events) >= 1

    def test_pr_url_in_tool_result(self, tmp_path: Path) -> None:
        """Detects PR URL in tool result."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry(
                "https://github.com/org/repo/pull/42\n"
                "PR created successfully."
            ),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_create"]
        assert len(pr_events) >= 1

    def test_gh_pr_create_in_result(self, tmp_path: Path) -> None:
        """Detects gh pr create in tool result content."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Running: gh pr create --title 'Fix' --body 'desc'"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_create"]
        assert len(pr_events) >= 1


# ---------------------------------------------------------------------------
# No milestones in clean JSONL
# ---------------------------------------------------------------------------


class TestNoMilestones:
    """Tests for JSONL with no milestone events."""

    def test_normal_tool_result(self, tmp_path: Path) -> None:
        """Normal tool output that is not a milestone."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("File content:\ndef hello():\n    print('hi')"),
            _tool_call_entry("Bash", "ls -la /tmp"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty JSONL file produces no milestones."""
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text("", encoding="utf-8")

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file produces no milestones."""
        jsonl = tmp_path / "nonexistent.jsonl"

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []


# ---------------------------------------------------------------------------
# Grace period suppression
# ---------------------------------------------------------------------------


class TestGracePeriod:
    """Tests for grace period suppression."""

    def test_within_grace_period_skips(self) -> None:
        """Milestone detected but within grace period -> skip."""
        trigger = MemoryFilingTrigger(grace_after_event_seconds=60)
        # Simulate a milestone just detected
        trigger._last_milestone_time = time.time()

        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "skip_grace_period"

    def test_grace_expired_fires(self) -> None:
        """Milestone detected and grace period expired -> fire."""
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=60, cooldown_seconds=0
        )
        # Simulate a milestone detected 120s ago
        trigger._last_milestone_time = time.time() - 120

        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "fire"

    def test_memory_filed_within_grace_suppresses(self, tmp_path: Path) -> None:
        """Milestone + memory filing within grace -> skip_memory_filed."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry(
                "[main abc1234] Add feature\n 1 file changed"
            ),
            _tool_call_entry("Bash", "memory new topic -d desc -c atlas --author kelvin -b body"),
        ])

        trigger = MemoryFilingTrigger(grace_after_event_seconds=60)
        milestones = trigger.scan_for_milestones(jsonl)

        assert len(milestones) >= 1
        # The scan also detected memory filing
        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "skip_memory_filed"

    def test_memory_update_within_grace_suppresses(self, tmp_path: Path) -> None:
        """Milestone + memory update within grace -> skip_memory_filed."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Merged pull request #10 into main"),
            _tool_call_entry("Bash", "memory update memory/kelvin/atlas/topic.md -b body"),
        ])

        trigger = MemoryFilingTrigger(grace_after_event_seconds=60)
        trigger.scan_for_milestones(jsonl)

        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "skip_memory_filed"


# ---------------------------------------------------------------------------
# Cooldown enforcement
# ---------------------------------------------------------------------------


class TestCooldown:
    """Tests for cooldown enforcement."""

    def test_cooldown_active_skips(self) -> None:
        """After a reminder, within cooldown -> skip."""
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=0, cooldown_seconds=120
        )
        trigger._last_milestone_time = time.time() - 200  # well past grace

        # Fire once
        d1 = trigger.evaluate(milestone_detected=True)
        assert d1.action == "fire"
        trigger.record_reminder_sent()

        # Reset memory-filed and pending for clean cooldown test
        trigger._memory_filed_since_milestone = False
        trigger._pending = False

        d2 = trigger.evaluate(milestone_detected=True)
        assert d2.action == "skip_cooldown"

    def test_cooldown_expired_fires(self) -> None:
        """After cooldown expires -> fire again."""
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=0, cooldown_seconds=120
        )
        trigger._last_milestone_time = time.time() - 200

        # Simulate reminder sent long ago
        trigger._last_reminder_time = time.time() - 200

        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "fire"

    def test_zero_cooldown_fires_immediately(self) -> None:
        """Cooldown of 0 -> fires immediately after previous reminder."""
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=0, cooldown_seconds=0
        )
        trigger._last_milestone_time = time.time() - 100

        d1 = trigger.evaluate(milestone_detected=True)
        assert d1.action == "fire"
        trigger.record_reminder_sent()

        # Reset for second evaluation
        trigger._memory_filed_since_milestone = False
        trigger._pending = False

        d2 = trigger.evaluate(milestone_detected=True)
        assert d2.action == "fire"


# ---------------------------------------------------------------------------
# Incremental scanning (from_position)
# ---------------------------------------------------------------------------


class TestIncrementalScanning:
    """Tests for incremental JSONL scanning."""

    def test_from_position_skips_earlier_content(self, tmp_path: Path) -> None:
        """Scanning from a position skips content before that offset."""
        jsonl = tmp_path / "session.jsonl"
        line1 = _tool_result_entry("Merged pull request #1 into main")
        line2 = _tool_result_entry("Regular output: ls -la")
        line3 = _tool_result_entry("Merged pull request #2 into main")

        _write_jsonl(jsonl, [line1, line2, line3])

        # First scan: finds the first merge
        trigger = MemoryFilingTrigger()
        events1 = trigger.scan_for_milestones(jsonl, from_position=0)
        pos1 = trigger.get_new_position()

        pr_events = [e for e in events1 if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1

        # Append a new line
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(_tool_result_entry("Merged pull request #3 into main") + "\n")

        # Second scan from previous position
        trigger2 = MemoryFilingTrigger()
        events2 = trigger2.scan_for_milestones(jsonl, from_position=pos1)

        pr_events2 = [e for e in events2 if e.event_type == "pr_merge"]
        assert len(pr_events2) >= 1

    def test_get_new_position_advances(self, tmp_path: Path) -> None:
        """get_new_position returns the byte offset after scanning."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Some output"),
        ])

        trigger = MemoryFilingTrigger()
        assert trigger.get_new_position() == 0

        trigger.scan_for_milestones(jsonl)
        pos = trigger.get_new_position()
        assert pos > 0

    def test_position_at_eof_returns_nothing(self, tmp_path: Path) -> None:
        """Scanning from EOF returns no events."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Merged pull request #1"),
        ])

        trigger = MemoryFilingTrigger()
        trigger.scan_for_milestones(jsonl)
        pos = trigger.get_new_position()

        # Scan from EOF
        events = trigger.scan_for_milestones(jsonl, from_position=pos)
        assert events == []


# ---------------------------------------------------------------------------
# Reminder format
# ---------------------------------------------------------------------------


class TestFormatReminder:
    """Tests for reminder message formatting."""

    def test_default_format(self) -> None:
        trigger = MemoryFilingTrigger()
        msg = trigger.format_reminder()

        assert "[ACP]" in msg
        assert "memory" in msg.lower()
        assert "filing" in msg.lower() or "file" in msg.lower() or "consider" in msg.lower()

    def test_contains_guidance(self) -> None:
        trigger = MemoryFilingTrigger()
        msg = trigger.format_reminder()

        # Should mention what kinds of things to file
        assert "takeaways" in msg or "decisions" in msg or "proverbs" in msg


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------


class TestCustomPatterns:
    """Tests for user-provided custom patterns."""

    def test_custom_pattern_in_result(self, tmp_path: Path) -> None:
        """Custom patterns detect events in tool results."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("DEPLOYMENT_COMPLETE: v2.1.0 shipped to production"),
        ])

        trigger = MemoryFilingTrigger(patterns=["DEPLOYMENT_COMPLETE"])
        events = trigger.scan_for_milestones(jsonl)

        custom_events = [e for e in events if e.event_type == "custom"]
        assert len(custom_events) >= 1

    def test_custom_pattern_in_tool_call(self, tmp_path: Path) -> None:
        """Custom patterns detect events in tool call inputs."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_call_entry("Bash", "deploy --env production --version 2.1.0"),
        ])

        trigger = MemoryFilingTrigger(patterns=["deploy --env production"])
        events = trigger.scan_for_milestones(jsonl)

        custom_events = [e for e in events if e.event_type == "custom"]
        assert len(custom_events) >= 1

    def test_custom_pattern_regex(self, tmp_path: Path) -> None:
        """Custom patterns support regex."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Release tag: v3.0.0-rc1 created"),
        ])

        trigger = MemoryFilingTrigger(patterns=[r"Release tag: v\d+\.\d+\.\d+"])
        events = trigger.scan_for_milestones(jsonl)

        custom_events = [e for e in events if e.event_type == "custom"]
        assert len(custom_events) >= 1

    def test_no_custom_patterns_by_default(self, tmp_path: Path) -> None:
        """Default trigger has no custom patterns."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("DEPLOYMENT_COMPLETE: shipped"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_malformed_json_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed JSONL lines are skipped gracefully."""
        jsonl = tmp_path / "session.jsonl"
        lines = [
            "not valid json at all",
            "{broken json",
            _tool_result_entry("Merged pull request #42"),
            "",  # empty line
            "null",
        ]
        _write_jsonl(jsonl, lines)

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1

    def test_entry_without_message(self, tmp_path: Path) -> None:
        """Entry with no message field is skipped."""
        jsonl = tmp_path / "session.jsonl"
        entry = json.dumps({"type": "user", "timestamp": "2026-02-26T12:00:00Z"})
        _write_jsonl(jsonl, [entry])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_entry_with_non_dict_message(self, tmp_path: Path) -> None:
        """Entry with non-dict message is skipped."""
        jsonl = tmp_path / "session.jsonl"
        entry = json.dumps({
            "type": "user",
            "timestamp": "2026-02-26T12:00:00Z",
            "message": "just a string",
        })
        _write_jsonl(jsonl, [entry])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_entry_with_non_list_content(self, tmp_path: Path) -> None:
        """Entry with non-list content is skipped."""
        jsonl = tmp_path / "session.jsonl"
        entry = json.dumps({
            "type": "user",
            "timestamp": "2026-02-26T12:00:00Z",
            "message": {"role": "user", "content": "just a string"},
        })
        _write_jsonl(jsonl, [entry])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_non_dict_entry(self, tmp_path: Path) -> None:
        """Non-dict parsed entry is skipped."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, ['"just a string"', "42", "true"])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_timestamp_non_string(self, tmp_path: Path) -> None:
        """Non-string timestamp is converted to string."""
        jsonl = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "timestamp": 1234567890,
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "toolu_test",
                        "type": "tool_result",
                        "content": "Merged pull request #5",
                    }
                ],
            },
        }
        _write_jsonl(jsonl, [json.dumps(entry)])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1
        assert pr_events[0].timestamp == "1234567890"

    def test_tool_result_with_non_string_content(self, tmp_path: Path) -> None:
        """Tool result with non-string content field is skipped."""
        jsonl = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "timestamp": "2026-02-26T12:00:00Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "toolu_test",
                        "type": "tool_result",
                        "content": 42,
                    }
                ],
            },
        }
        _write_jsonl(jsonl, [json.dumps(entry)])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_tool_use_with_non_dict_input(self, tmp_path: Path) -> None:
        """Tool use with non-dict input is skipped."""
        jsonl = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "timestamp": "2026-02-26T12:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_test",
                        "name": "Bash",
                        "input": "not a dict",
                    }
                ],
            },
        }
        _write_jsonl(jsonl, [json.dumps(entry)])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_os_error_during_read(self, tmp_path: Path) -> None:
        """OS error during file read returns empty list."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Merged PR #42"),
        ])

        trigger = MemoryFilingTrigger()

        with patch.object(Path, "open", side_effect=OSError("Permission denied")):
            events = trigger.scan_for_milestones(jsonl)

        assert events == []

    def test_detail_truncation(self, tmp_path: Path) -> None:
        """Long tool output is truncated in event details."""
        long_content = "Merged " + "x" * 300
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry(long_content),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        pr_events = [e for e in events if e.event_type == "pr_merge"]
        assert len(pr_events) >= 1
        assert len(pr_events[0].details) <= 124  # 120 + "..."

    def test_no_milestone_no_last_time_evaluates_skip(self) -> None:
        """Evaluate with no milestones and no previous milestone -> skip."""
        trigger = MemoryFilingTrigger()
        decision = trigger.evaluate(milestone_detected=False)
        assert decision.action == "skip_no_milestone"

    def test_pending_reminder_skips(self) -> None:
        """Pending reminder prevents duplicate firing."""
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=0, cooldown_seconds=0
        )
        trigger._last_milestone_time = time.time() - 100

        d1 = trigger.evaluate(milestone_detected=True)
        assert d1.action == "fire"
        trigger.record_reminder_sent()

        # Still pending, memory not filed
        trigger._memory_filed_since_milestone = False

        d2 = trigger.evaluate(milestone_detected=True)
        assert d2.action == "skip_pending"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    """Tests for record_reminder_sent, record_memory_filed, reset."""

    def test_record_reminder_sent(self) -> None:
        trigger = MemoryFilingTrigger()
        assert trigger._pending is False
        assert trigger._last_reminder_time is None

        before = time.time()
        trigger.record_reminder_sent()
        after = time.time()

        assert trigger._pending is True
        assert trigger._last_reminder_time is not None
        assert before <= trigger._last_reminder_time <= after

    def test_record_memory_filed(self) -> None:
        trigger = MemoryFilingTrigger()
        trigger._pending = True
        trigger._memory_filed_since_milestone = False

        trigger.record_memory_filed()

        assert trigger._memory_filed_since_milestone is True
        assert trigger._pending is False

    def test_reset_for_new_milestone(self) -> None:
        trigger = MemoryFilingTrigger()
        trigger._memory_filed_since_milestone = True
        trigger._last_milestone_time = time.time()
        trigger._pending = True

        trigger.reset_for_new_milestone()

        assert trigger._memory_filed_since_milestone is False
        assert trigger._last_milestone_time is None
        assert trigger._pending is False


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestConstructorDefaults:
    """Tests for constructor parameter defaults."""

    def test_default_grace(self) -> None:
        trigger = MemoryFilingTrigger()
        assert trigger.grace_after_event_seconds == 60

    def test_default_cooldown(self) -> None:
        trigger = MemoryFilingTrigger()
        assert trigger.cooldown_seconds == 120

    def test_custom_values(self) -> None:
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=30,
            cooldown_seconds=60,
            patterns=["CUSTOM_PATTERN"],
        )
        assert trigger.grace_after_event_seconds == 30
        assert trigger.cooldown_seconds == 60
        assert len(trigger._custom_patterns) == 1

    def test_initial_state_clean(self) -> None:
        trigger = MemoryFilingTrigger()
        assert trigger._last_reminder_time is None
        assert trigger._pending is False
        assert trigger._last_milestone_time is None
        assert trigger._memory_filed_since_milestone is False
        assert trigger._last_position == 0


# ---------------------------------------------------------------------------
# TriggerDecision reuse
# ---------------------------------------------------------------------------


class TestTriggerDecisionReuse:
    """Verify MemoryFilingTrigger reuses TriggerDecision from compaction."""

    def test_evaluate_returns_trigger_decision(self) -> None:
        trigger = MemoryFilingTrigger()
        decision = trigger.evaluate(milestone_detected=False)
        assert isinstance(decision, TriggerDecision)

    def test_decision_fields_populated(self) -> None:
        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=0, cooldown_seconds=0
        )
        trigger._last_milestone_time = time.time() - 100

        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "fire"
        assert len(decision.reason) > 0
        # Tokens/threshold are 0 for memory trigger
        assert decision.current_tokens == 0
        assert decision.threshold == 0


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """Integration-style tests combining scanning, evaluation, and state."""

    def test_milestone_grace_fire_cycle(self, tmp_path: Path) -> None:
        """Full cycle: scan -> grace -> fire -> record -> cooldown."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Merged pull request #42 into main"),
        ])

        trigger = MemoryFilingTrigger(
            grace_after_event_seconds=0, cooldown_seconds=120
        )

        # Scan and detect milestone
        events = trigger.scan_for_milestones(jsonl)
        assert len(events) >= 1

        # Evaluate -> should fire (grace=0)
        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "fire"

        # Record reminder sent
        trigger.record_reminder_sent()

        # Reset memory-filed for next eval
        trigger._memory_filed_since_milestone = False

        # Try again -> cooldown
        trigger._pending = False
        d2 = trigger.evaluate(milestone_detected=True)
        assert d2.action == "skip_cooldown"

    def test_milestone_then_memory_filing_suppresses(self, tmp_path: Path) -> None:
        """Agent files memory after milestone -> reminder suppressed."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_result_entry("Merged pull request #42 into main"),
            _tool_call_entry("Bash", "memory new topic -d desc -c atlas --author kelvin -b body"),
        ])

        trigger = MemoryFilingTrigger(grace_after_event_seconds=60)
        events = trigger.scan_for_milestones(jsonl)
        assert len(events) >= 1

        # Should skip because memory was filed
        decision = trigger.evaluate(milestone_detected=True)
        assert decision.action == "skip_memory_filed"

    def test_multiple_milestones_in_one_scan(self, tmp_path: Path) -> None:
        """Multiple different milestones in a single scan."""
        jsonl = tmp_path / "session.jsonl"
        _write_jsonl(jsonl, [
            _tool_call_entry("Bash", "git commit -m 'Fix bug'"),
            _tool_result_entry("[main abc1234] Fix bug\n 1 file changed"),
            _tool_call_entry("Bash", "gh pr create --title 'Fix bug'"),
            _tool_result_entry("https://github.com/org/repo/pull/99"),
        ])

        trigger = MemoryFilingTrigger()
        events = trigger.scan_for_milestones(jsonl)

        event_types = {e.event_type for e in events}
        assert "commit" in event_types
        assert "pr_create" in event_types
