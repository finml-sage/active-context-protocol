"""Tests for the tmux Delivery System."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.delivery import DeliveryResult, DeliverySystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    """Return a path for the audit log file."""
    return tmp_path / "delivery.log"


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    """Return a path for a fake JSONL session file."""
    return tmp_path / "session.jsonl"


@pytest.fixture
def ds(tmp_log: Path) -> DeliverySystem:
    """DeliverySystem with a short warmdown for tests."""
    return DeliverySystem(
        tmux_session="test-session",
        warmdown_seconds=2,
        idle_threshold_seconds=0.5,
        log_file=tmp_log,
    )


def _write_jsonl(path: Path, entries: list[dict], mtime_offset: float = -10) -> None:
    """Write JSONL entries to a file and set mtime to the past."""
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")
    # Set mtime in the past so idle check passes
    now = time.time()
    import os

    os.utime(path, (now + mtime_offset, now + mtime_offset))


# ---------------------------------------------------------------------------
# DeliveryResult tests
# ---------------------------------------------------------------------------


class TestDeliveryResult:
    def test_dataclass_fields(self) -> None:
        now = datetime.now(timezone.utc)
        result = DeliveryResult(
            success=True,
            outcome="delivered",
            timestamp=now,
            trigger_type="compaction",
            message="/compact test",
        )
        assert result.success is True
        assert result.outcome == "delivered"
        assert result.trigger_type == "compaction"
        assert result.message == "/compact test"
        assert result.timestamp == now

    def test_failed_result(self) -> None:
        result = DeliveryResult(
            success=False,
            outcome="failed",
            timestamp=datetime.now(timezone.utc),
            trigger_type="memory",
            message="reminder",
        )
        assert result.success is False
        assert result.outcome == "failed"


# ---------------------------------------------------------------------------
# Idle detection tests
# ---------------------------------------------------------------------------


class TestIdleDetection:
    def test_idle_when_stale_and_assistant(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Agent is idle: JSONL is old AND last entry is assistant."""
        _write_jsonl(
            tmp_jsonl,
            [
                {"type": "tool_use", "content": "running"},
                {"type": "assistant", "content": "done"},
            ],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is True

    def test_not_idle_when_recent_mtime(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """JSONL was just written — agent is active."""
        _write_jsonl(
            tmp_jsonl,
            [{"type": "assistant", "content": "done"}],
            mtime_offset=0,  # mtime is now
        )
        assert ds.is_idle(tmp_jsonl) is False

    def test_not_idle_when_tool_use(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Last entry is tool_use — agent is mid-processing."""
        _write_jsonl(
            tmp_jsonl,
            [{"type": "tool_use", "content": "running command"}],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is False

    def test_not_idle_when_progress(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Last entry is progress — agent is mid-processing."""
        _write_jsonl(
            tmp_jsonl,
            [{"type": "progress", "content": "thinking"}],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is False

    def test_not_idle_when_file_missing(self, ds: DeliverySystem) -> None:
        """Non-existent JSONL file — err on side of not delivering."""
        missing = Path("/tmp/nonexistent_jsonl_file_12345.jsonl")
        assert ds.is_idle(missing) is False

    def test_not_idle_when_empty_file(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Empty JSONL file — no entries to check."""
        tmp_jsonl.write_text("")
        assert ds.is_idle(tmp_jsonl) is False

    def test_not_idle_when_invalid_json(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Malformed JSON in last line — err on side of not delivering."""
        tmp_jsonl.write_text("not valid json\n")
        import os

        os.utime(tmp_jsonl, (time.time() - 10, time.time() - 10))
        assert ds.is_idle(tmp_jsonl) is False

    def test_not_idle_when_no_type_field(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """JSON entry without 'type' field — not recognizable as assistant."""
        _write_jsonl(
            tmp_jsonl,
            [{"content": "hello"}],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is False

    def test_not_idle_when_assistant_has_tool_use_blocks(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Assistant entry with tool_use content blocks = tool call initiated."""
        _write_jsonl(
            tmp_jsonl,
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "toolu_xxx", "name": "Bash"}
                        ]
                    },
                }
            ],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is False

    def test_idle_when_assistant_has_text_only(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Assistant entry with only text content = agent finished turn."""
        _write_jsonl(
            tmp_jsonl,
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Here is my response."}
                        ]
                    },
                }
            ],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is True

    def test_idle_with_many_entries(
        self, ds: DeliverySystem, tmp_jsonl: Path
    ) -> None:
        """Only the LAST entry matters for idle detection."""
        entries = [
            {"type": "tool_use", "content": "step 1"},
            {"type": "progress", "content": "step 2"},
            {"type": "tool_use", "content": "step 3"},
            {"type": "assistant", "content": "final response"},
        ]
        _write_jsonl(tmp_jsonl, entries, mtime_offset=-10)
        assert ds.is_idle(tmp_jsonl) is True

    def test_configurable_idle_threshold(
        self, tmp_jsonl: Path, tmp_log: Path
    ) -> None:
        """Custom idle threshold is respected."""
        ds_long = DeliverySystem(
            tmux_session="test",
            idle_threshold_seconds=60.0,
            log_file=tmp_log,
        )
        # File mtime 10 seconds ago — below 60-second threshold
        _write_jsonl(
            tmp_jsonl,
            [{"type": "assistant", "content": "done"}],
            mtime_offset=-10,
        )
        assert ds_long.is_idle(tmp_jsonl) is False


# ---------------------------------------------------------------------------
# Warmdown tests
# ---------------------------------------------------------------------------


class TestWarmdown:
    def test_can_deliver_initially(self, ds: DeliverySystem) -> None:
        """No prior delivery — warmdown does not block."""
        assert ds.can_deliver() is True

    def test_warmdown_blocks_after_delivery(self, ds: DeliverySystem) -> None:
        """Immediately after setting last_delivery_time, warmdown blocks."""
        ds._last_delivery_time = datetime.now(timezone.utc)
        assert ds.can_deliver() is False

    def test_warmdown_expires(self, tmp_log: Path) -> None:
        """After warmdown_seconds, delivery is allowed again."""
        ds = DeliverySystem(
            tmux_session="test",
            warmdown_seconds=0,  # Immediate expiry for testing
            log_file=tmp_log,
        )
        ds._last_delivery_time = datetime.now(timezone.utc)
        assert ds.can_deliver() is True


# ---------------------------------------------------------------------------
# tmux availability / session checks
# ---------------------------------------------------------------------------


class TestTmuxChecks:
    @patch("src.delivery.shutil.which", return_value=None)
    def test_no_tmux_binary(self, mock_which: MagicMock, ds: DeliverySystem) -> None:
        """tmux not installed — skip gracefully."""
        result = ds.deliver("test message", "compaction")
        assert result.success is False
        assert result.outcome == "skipped_no_tmux"

    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_no_session(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """Target session doesn't exist — skip gracefully."""
        # has-session returns non-zero
        mock_run.return_value = MagicMock(returncode=1)
        result = ds.deliver("test message", "compaction")
        assert result.success is False
        assert result.outcome == "skipped_no_session"


# ---------------------------------------------------------------------------
# Successful delivery tests
# ---------------------------------------------------------------------------


class TestDelivery:
    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_successful_delivery(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """Happy path: tmux available, session exists, warmdown clear."""
        # has-session succeeds, send-keys calls succeed
        mock_run.return_value = MagicMock(returncode=0)
        result = ds.deliver("/compact test", "compaction")

        assert result.success is True
        assert result.outcome == "delivered"
        assert result.trigger_type == "compaction"
        assert result.message == "/compact test"

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_two_call_tmux_pattern(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """Verify the two-call tmux pattern: text first, then Enter."""
        mock_run.return_value = MagicMock(returncode=0)
        ds.deliver("/compact test", "compaction")

        # Should be 3 calls: has-session, send-keys (text), send-keys (Enter)
        assert mock_run.call_count == 3

        # Call 0: has-session check
        args_0 = mock_run.call_args_list[0]
        assert args_0[0][0] == ["tmux", "has-session", "-t", "test-session"]

        # Call 1: send the message text
        args_1 = mock_run.call_args_list[1]
        assert args_1[0][0] == [
            "tmux",
            "send-keys",
            "-t",
            "test-session",
            "/compact test",
        ]
        assert args_1[1].get("check") is True

        # Call 2: send Enter separately
        args_2 = mock_run.call_args_list[2]
        assert args_2[0][0] == [
            "tmux",
            "send-keys",
            "-t",
            "test-session",
            "Enter",
        ]
        assert args_2[1].get("check") is True

        # Sleep between the two send-keys calls
        mock_sleep.assert_called_once_with(0.5)

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_warmdown_after_delivery(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """After a successful delivery, warmdown blocks the next one."""
        mock_run.return_value = MagicMock(returncode=0)

        result1 = ds.deliver("msg1", "compaction")
        assert result1.success is True

        result2 = ds.deliver("msg2", "memory")
        assert result2.success is False
        assert result2.outcome == "queued_warmdown"

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_delivery_failure_subprocess_error(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """subprocess error during send-keys -> failed outcome."""
        import subprocess as sp

        # has-session succeeds, but send-keys raises
        def side_effect(args, **kwargs):
            if "has-session" in args:
                return MagicMock(returncode=0)
            raise sp.CalledProcessError(1, args)

        mock_run.side_effect = side_effect
        result = ds.deliver("test", "compaction")
        assert result.success is False
        assert result.outcome == "failed"


# ---------------------------------------------------------------------------
# Audit trail tests
# ---------------------------------------------------------------------------


class TestAuditTrail:
    @patch("src.delivery.shutil.which", return_value=None)
    def test_audit_log_written(
        self, mock_which: MagicMock, tmp_log: Path
    ) -> None:
        """Every delivery attempt produces an audit trail entry."""
        ds = DeliverySystem(
            tmux_session="test",
            log_file=tmp_log,
        )
        ds.deliver("hello world", "test_trigger")

        log_content = tmp_log.read_text()
        assert "test_trigger" in log_content
        assert "skipped_no_tmux" in log_content
        assert "hello world" in log_content
        assert "mode=reminder" in log_content

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_audit_log_success(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        tmp_log: Path,
    ) -> None:
        """Successful delivery is logged too."""
        mock_run.return_value = MagicMock(returncode=0)
        ds = DeliverySystem(tmux_session="test", log_file=tmp_log)
        ds.deliver("/compact now", "compaction")

        log_content = tmp_log.read_text()
        assert "delivered" in log_content
        assert "compaction" in log_content

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_audit_multiple_entries(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        tmp_log: Path,
    ) -> None:
        """Multiple attempts produce multiple audit lines."""
        mock_run.return_value = MagicMock(returncode=0)
        ds = DeliverySystem(
            tmux_session="test",
            warmdown_seconds=0,
            log_file=tmp_log,
        )
        ds.deliver("msg1", "trigger1")
        ds.deliver("msg2", "trigger2")

        lines = [l for l in tmp_log.read_text().strip().splitlines() if l]
        assert len(lines) == 2

    def test_audit_to_stderr_when_no_logfile(self) -> None:
        """When no log_file, audit goes to stderr (no crash)."""
        ds = DeliverySystem(tmux_session="test", log_file=None)
        # Just verifying construction doesn't raise
        assert ds._audit_logger is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_multiline_message(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """Multiline messages are sent as-is (tmux handles them)."""
        mock_run.return_value = MagicMock(returncode=0)
        msg = "/compact Retain: tasks\nCompress: history"
        result = ds.deliver(msg, "compaction")
        assert result.success is True

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_last_delivery_time_updated_on_success(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """_last_delivery_time is set after successful delivery."""
        mock_run.return_value = MagicMock(returncode=0)
        assert ds._last_delivery_time is None
        ds.deliver("test", "compaction")
        assert ds._last_delivery_time is not None

    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_last_delivery_time_not_updated_on_failure(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """_last_delivery_time is NOT set when delivery fails."""
        import subprocess as sp

        def side_effect(args, **kwargs):
            if "has-session" in args:
                return MagicMock(returncode=0)
            raise sp.CalledProcessError(1, args)

        mock_run.side_effect = side_effect
        ds.deliver("test", "compaction")
        assert ds._last_delivery_time is None

    def test_read_last_line_large_file(self, tmp_path: Path) -> None:
        """_read_last_line works on files larger than 8KB."""
        path = tmp_path / "large.jsonl"
        # Write >8KB of data
        lines = [json.dumps({"type": "tool_use", "i": i}) for i in range(200)]
        lines.append(json.dumps({"type": "assistant", "content": "final"}))
        path.write_text("\n".join(lines) + "\n")
        result = DeliverySystem._read_last_line(path)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["type"] == "assistant"

    def test_read_last_line_single_line(self, tmp_path: Path) -> None:
        """_read_last_line works with a single-line file."""
        path = tmp_path / "single.jsonl"
        path.write_text(json.dumps({"type": "assistant"}) + "\n")
        result = DeliverySystem._read_last_line(path)
        assert result is not None
        assert json.loads(result)["type"] == "assistant"

    def test_read_last_line_empty_file(self, tmp_path: Path) -> None:
        """_read_last_line returns None for empty files."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        # stat().st_size == 0 -> returns None
        result = DeliverySystem._read_last_line(path)
        assert result is None


# ---------------------------------------------------------------------------
# Delivery mode tests (Issue #13)
# ---------------------------------------------------------------------------


class TestDeliveryModes:
    """Tests for the reminder vs command delivery mode behavior.

    Issue #13: Idle detection was blocking all reminder delivery during
    active sessions.  The fix introduces delivery modes so that reminders
    bypass idle detection while commands still require it.
    """

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_reminder_mode_skips_idle_check(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
        tmp_jsonl: Path,
    ) -> None:
        """Reminder mode delivers even when the agent is actively working."""
        mock_run.return_value = MagicMock(returncode=0)

        # Write an active (non-idle) JSONL — mtime is NOW, last entry is tool_use
        _write_jsonl(
            tmp_jsonl,
            [{"type": "tool_use", "content": "running command"}],
            mtime_offset=0,
        )
        # Verify idle detection would return False
        assert ds.is_idle(tmp_jsonl) is False

        # Deliver in reminder mode — should succeed anyway
        result = ds.deliver(
            "Time to save memory", "memory_filing", mode="reminder"
        )
        assert result.success is True
        assert result.outcome == "delivered"

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_command_mode_requires_idle(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
        tmp_jsonl: Path,
    ) -> None:
        """Command mode blocks delivery when the agent is not idle."""
        mock_run.return_value = MagicMock(returncode=0)

        # Write an active (non-idle) JSONL
        _write_jsonl(
            tmp_jsonl,
            [{"type": "tool_use", "content": "running command"}],
            mtime_offset=0,
        )

        result = ds.deliver(
            "/compact now", "compaction", mode="command", jsonl_path=tmp_jsonl
        )
        assert result.success is False
        assert result.outcome == "queued_not_idle"

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_command_mode_delivers_when_idle(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
        tmp_jsonl: Path,
    ) -> None:
        """Command mode delivers when the agent IS idle."""
        mock_run.return_value = MagicMock(returncode=0)

        # Write an idle JSONL — old mtime, last entry is assistant text
        _write_jsonl(
            tmp_jsonl,
            [{"type": "assistant", "content": "done"}],
            mtime_offset=-10,
        )
        assert ds.is_idle(tmp_jsonl) is True

        result = ds.deliver(
            "/compact now", "compaction", mode="command", jsonl_path=tmp_jsonl
        )
        assert result.success is True
        assert result.outcome == "delivered"

    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_command_mode_fails_without_jsonl_path(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """Command mode with no jsonl_path fails gracefully."""
        mock_run.return_value = MagicMock(returncode=0)

        result = ds.deliver(
            "/compact now", "compaction", mode="command", jsonl_path=None
        )
        assert result.success is False
        assert result.outcome == "failed"

    def test_default_mode_is_reminder(self, ds: DeliverySystem) -> None:
        """deliver() defaults to reminder mode for backwards compatibility."""
        # Verify by calling with only positional args (no mode kwarg).
        # We only need to check the signature default, not actual delivery.
        import inspect

        sig = inspect.signature(ds.deliver)
        mode_param = sig.parameters["mode"]
        assert mode_param.default == "reminder"

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_warmdown_enforced_in_reminder_mode(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        ds: DeliverySystem,
    ) -> None:
        """Warmdown is still enforced even in reminder mode."""
        mock_run.return_value = MagicMock(returncode=0)

        result1 = ds.deliver("msg1", "compaction", mode="reminder")
        assert result1.success is True

        result2 = ds.deliver("msg2", "memory_filing", mode="reminder")
        assert result2.success is False
        assert result2.outcome == "queued_warmdown"

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_audit_log_records_reminder_mode(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        tmp_log: Path,
    ) -> None:
        """Audit trail includes mode=reminder for reminder deliveries."""
        mock_run.return_value = MagicMock(returncode=0)
        ds = DeliverySystem(tmux_session="test", log_file=tmp_log)
        ds.deliver("reminder text", "memory_filing", mode="reminder")

        log_content = tmp_log.read_text()
        assert "mode=reminder" in log_content
        assert "delivered" in log_content

    @patch("src.delivery.time.sleep")
    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_audit_log_records_command_mode(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        mock_sleep: MagicMock,
        tmp_log: Path,
        tmp_jsonl: Path,
    ) -> None:
        """Audit trail includes mode=command for command deliveries."""
        mock_run.return_value = MagicMock(returncode=0)
        ds = DeliverySystem(
            tmux_session="test",
            idle_threshold_seconds=0.5,
            log_file=tmp_log,
        )
        _write_jsonl(
            tmp_jsonl,
            [{"type": "assistant", "content": "done"}],
            mtime_offset=-10,
        )
        ds.deliver(
            "/compact now", "compaction", mode="command", jsonl_path=tmp_jsonl
        )

        log_content = tmp_log.read_text()
        assert "mode=command" in log_content
        assert "delivered" in log_content

    @patch("src.delivery.subprocess.run")
    @patch("src.delivery.shutil.which", return_value="/usr/bin/tmux")
    def test_audit_log_records_mode_on_not_idle(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        tmp_log: Path,
        tmp_jsonl: Path,
    ) -> None:
        """Audit trail records mode=command even when queued_not_idle."""
        mock_run.return_value = MagicMock(returncode=0)
        ds = DeliverySystem(
            tmux_session="test",
            idle_threshold_seconds=0.5,
            log_file=tmp_log,
        )
        # Active JSONL — not idle
        _write_jsonl(
            tmp_jsonl,
            [{"type": "tool_use", "content": "running"}],
            mtime_offset=0,
        )
        ds.deliver(
            "/compact now", "compaction", mode="command", jsonl_path=tmp_jsonl
        )

        log_content = tmp_log.read_text()
        assert "mode=command" in log_content
        assert "queued_not_idle" in log_content
