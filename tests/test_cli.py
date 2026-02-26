"""Tests for the ACP CLI module."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli import (
    _is_process_running,
    _read_pid,
    _remove_pid,
    _write_pid,
    build_parser,
    cmd_config,
    cmd_init,
    cmd_start,
    cmd_status,
    cmd_stop,
    main,
    run_monitor,
)
from src.config import AcpConfig, _DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect PID file to a temp directory."""
    pid_path = tmp_path / "acp.pid"
    monkeypatch.setattr("src.cli._PID_FILE", pid_path)
    return pid_path


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Return a path for a temporary config file."""
    return tmp_path / "config.yaml"


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------


class TestArgumentParser:
    def test_build_parser_returns_argument_parser(self) -> None:
        """build_parser returns an argparse.ArgumentParser instance."""
        import argparse

        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_start_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["start"])
        assert args.command == "start"

    def test_stop_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stop"])
        assert args.command == "stop"

    def test_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_config_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["config"])
        assert args.command == "config"

    def test_init_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.command == "init"
        assert args.force is False

    def test_init_with_force(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["init", "--force"])
        assert args.force is True

    def test_init_with_force_short(self) -> None:
        """The -f short form works for --force."""
        parser = build_parser()
        args = parser.parse_args(["init", "-f"])
        assert args.force is True

    def test_global_config_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--config", "/path/to/config.yaml", "status"])
        assert args.config == "/path/to/config.yaml"
        assert args.command == "status"

    def test_global_config_short_form(self) -> None:
        """The -c short form works for --config."""
        parser = build_parser()
        args = parser.parse_args(["-c", "/some/path.yaml", "config"])
        assert args.config == "/some/path.yaml"
        assert args.command == "config"

    def test_no_command_returns_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_config_default_is_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.config is None


# ---------------------------------------------------------------------------
# PID file management tests
# ---------------------------------------------------------------------------


class TestPidFile:
    def test_write_and_read_pid(self, tmp_pid: Path) -> None:
        _write_pid(12345)
        assert _read_pid() == 12345

    def test_write_pid_default_uses_current_process(self, tmp_pid: Path) -> None:
        """Calling _write_pid() with no argument writes os.getpid()."""
        _write_pid()
        assert _read_pid() == os.getpid()

    def test_read_pid_missing_file(self, tmp_pid: Path) -> None:
        assert _read_pid() is None

    def test_read_pid_invalid_content(self, tmp_pid: Path) -> None:
        """A PID file with non-numeric content returns None."""
        tmp_pid.parent.mkdir(parents=True, exist_ok=True)
        tmp_pid.write_text("not_a_number", encoding="utf-8")
        assert _read_pid() is None

    def test_remove_pid(self, tmp_pid: Path) -> None:
        _write_pid(12345)
        assert tmp_pid.is_file()
        _remove_pid()
        assert not tmp_pid.is_file()

    def test_remove_pid_missing_file(self, tmp_pid: Path) -> None:
        """Removing a non-existent PID file does not raise."""
        _remove_pid()  # Should not raise

    def test_write_pid_creates_parent_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pid_path = tmp_path / "deep" / "nested" / "acp.pid"
        monkeypatch.setattr("src.cli._PID_FILE", pid_path)
        _write_pid(99999)
        assert pid_path.is_file()
        assert _read_pid() == 99999

    def test_write_pid_returns_resolved_path(self, tmp_pid: Path) -> None:
        """_write_pid returns the resolved path of the PID file."""
        result = _write_pid(42)
        assert isinstance(result, Path)
        assert result.is_file()

    def test_is_process_running_self(self) -> None:
        """Our own PID should be running."""
        assert _is_process_running(os.getpid()) is True

    def test_is_process_running_invalid_pid(self) -> None:
        """A very high PID should not be running."""
        assert _is_process_running(999999999) is False

    def test_is_process_running_os_error(self) -> None:
        """OSError from os.kill returns False."""
        with patch("os.kill", side_effect=OSError("No such process")):
            assert _is_process_running(12345) is False

    def test_is_process_running_process_lookup_error(self) -> None:
        """ProcessLookupError from os.kill returns False."""
        with patch("os.kill", side_effect=ProcessLookupError("No such process")):
            assert _is_process_running(12345) is False


# ---------------------------------------------------------------------------
# cmd_init tests
# ---------------------------------------------------------------------------


class TestCmdInit:
    def test_init_creates_config(self, tmp_config: Path) -> None:
        """acp init creates the config file."""
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "init"])
        result = cmd_init(args)
        assert result == 0
        assert tmp_config.is_file()

    def test_init_refuses_overwrite(self, tmp_config: Path) -> None:
        """acp init refuses to overwrite existing config without --force."""
        tmp_config.write_text("existing content")
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "init"])
        result = cmd_init(args)
        assert result == 1
        assert tmp_config.read_text() == "existing content"

    def test_init_force_overwrite(self, tmp_config: Path) -> None:
        """acp init --force overwrites existing config."""
        tmp_config.write_text("existing content")
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "init", "--force"])
        result = cmd_init(args)
        assert result == 0
        assert "Active Context Protocol" in tmp_config.read_text()

    def test_init_prints_path(self, tmp_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """acp init prints the path of the created file."""
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "init"])
        cmd_init(args)
        captured = capsys.readouterr()
        assert str(tmp_config) in captured.out

    def test_init_refuse_message_mentions_force(
        self, tmp_config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When refusing overwrite, the message tells the user about --force."""
        tmp_config.write_text("content")
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "init"])
        cmd_init(args)
        captured = capsys.readouterr()
        assert "--force" in captured.out


# ---------------------------------------------------------------------------
# cmd_status tests
# ---------------------------------------------------------------------------


class TestCmdStatus:
    def test_status_when_not_running(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp status when no monitor is running."""
        parser = build_parser()
        args = parser.parse_args(["status"])
        result = cmd_status(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "not running" in captured.out.lower()

    def test_status_when_running(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp status when a monitor PID exists and is alive."""
        _write_pid(os.getpid())  # Use our own PID (it's running)
        parser = build_parser()
        args = parser.parse_args(["status"])
        result = cmd_status(args)
        captured = capsys.readouterr()
        assert "running" in captured.out.lower()
        assert str(os.getpid()) in captured.out

    def test_status_with_stale_pid(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp status when PID file exists but process is dead."""
        _write_pid(999999999)  # Non-existent PID
        parser = build_parser()
        args = parser.parse_args(["status"])
        result = cmd_status(args)
        captured = capsys.readouterr()
        assert "not running" in captured.out.lower()

    def test_status_shows_config_path(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp status output includes the config path."""
        parser = build_parser()
        args = parser.parse_args(["status"])
        cmd_status(args)
        captured = capsys.readouterr()
        assert "Config:" in captured.out

    def test_status_shows_pid_file_path(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp status output includes the PID file path."""
        parser = build_parser()
        args = parser.parse_args(["status"])
        cmd_status(args)
        captured = capsys.readouterr()
        assert "PID file:" in captured.out


# ---------------------------------------------------------------------------
# cmd_config tests
# ---------------------------------------------------------------------------


class TestCmdConfig:
    def test_config_shows_defaults(self, capsys: pytest.CaptureFixture[str]) -> None:
        """acp config displays all resolved settings."""
        parser = build_parser()
        args = parser.parse_args(["config"])
        result = cmd_config(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "token_threshold" in captured.out
        assert "70000" in captured.out
        assert "polling_interval" in captured.out

    def test_config_with_custom_file(
        self, tmp_config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp config with a custom config file."""
        tmp_config.write_text("token_threshold: 99999\n")
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "config"])
        result = cmd_config(args)
        captured = capsys.readouterr()
        assert "99999" in captured.out

    def test_config_prints_all_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        """acp config output includes every field name from AcpConfig."""
        from dataclasses import fields as dc_fields

        parser = build_parser()
        args = parser.parse_args(["config"])
        cmd_config(args)
        captured = capsys.readouterr()
        for f in dc_fields(AcpConfig):
            assert f.name in captured.out, f"Missing field {f.name} in config output"

    def test_config_prints_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        """acp config output has a header line."""
        parser = build_parser()
        args = parser.parse_args(["config"])
        cmd_config(args)
        captured = capsys.readouterr()
        assert "Resolved ACP configuration" in captured.out


# ---------------------------------------------------------------------------
# cmd_stop tests
# ---------------------------------------------------------------------------


class TestCmdStop:
    def test_stop_no_pid_file(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp stop when no PID file exists."""
        result = cmd_stop(build_parser().parse_args(["stop"]))
        assert result == 1
        captured = capsys.readouterr()
        assert "no pid file" in captured.out.lower()

    def test_stop_stale_pid(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp stop when PID exists but process is dead."""
        _write_pid(999999999)
        result = cmd_stop(build_parser().parse_args(["stop"]))
        assert result == 1
        captured = capsys.readouterr()
        assert "not running" in captured.out.lower()
        # Stale PID file should be cleaned up
        assert not tmp_pid.is_file()

    def test_stop_running_pid_sends_sigterm(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp stop sends SIGTERM to a running process and cleans up PID file."""
        _write_pid(54321)
        with (
            patch("src.cli._is_process_running", return_value=True),
            patch("src.cli.os.kill") as mock_kill,
        ):
            result = cmd_stop(build_parser().parse_args(["stop"]))
        assert result == 0
        mock_kill.assert_called_once_with(54321, signal.SIGTERM)
        captured = capsys.readouterr()
        assert "SIGTERM" in captured.out
        assert "54321" in captured.out

    def test_stop_os_error_on_kill(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp stop returns 1 if os.kill raises OSError."""
        _write_pid(54321)
        with (
            patch("src.cli._is_process_running", return_value=True),
            patch("src.cli.os.kill", side_effect=OSError("Operation not permitted")),
        ):
            result = cmd_stop(build_parser().parse_args(["stop"]))
        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to stop" in captured.out


# ---------------------------------------------------------------------------
# cmd_start tests
# ---------------------------------------------------------------------------


class TestCmdStart:
    def test_start_refuses_if_already_running(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp start when a monitor is already running."""
        _write_pid(os.getpid())  # Our PID is running
        parser = build_parser()
        args = parser.parse_args(["start"])
        result = cmd_start(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "already running" in captured.out.lower()

    def test_start_with_no_tmux_session_warns(
        self, tmp_pid: Path, tmp_config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp start with no tmux_session configured prints a warning."""
        tmp_config.write_text("tmux_session: \"\"\n")
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "start"])
        with patch("src.cli.run_monitor"):
            result = cmd_start(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_start_with_tmux_session_shows_session(
        self, tmp_pid: Path, tmp_config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """acp start with tmux_session configured prints the session name."""
        tmp_config.write_text("tmux_session: my-session\n")
        parser = build_parser()
        args = parser.parse_args(["--config", str(tmp_config), "start"])
        with patch("src.cli.run_monitor"):
            result = cmd_start(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "my-session" in captured.out


# ---------------------------------------------------------------------------
# Main loop logic tests (mocked components)
# ---------------------------------------------------------------------------


class TestRunMonitor:
    def test_polling_cycle_no_session(self, tmp_pid: Path) -> None:
        """Monitor loop handles no active session gracefully."""
        config = AcpConfig(polling_interval=1, tmux_session="test")

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor"),
            patch("src.cli.DeliverySystem"),
            patch("src.cli.CompactionTrigger"),
            patch("src.cli.MemoryFilingTrigger"),
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker_instance = MockTracker.return_value
            tracker_instance.find_active_session.return_value = None

            # Make sleep raise to break the loop after one cycle
            call_count = 0

            def sleep_side_effect(seconds: float) -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise KeyboardInterrupt

            mock_sleep.side_effect = sleep_side_effect

            run_monitor(config)

            tracker_instance.find_active_session.assert_called()
            assert call_count >= 1

    def test_polling_cycle_with_session(self, tmp_pid: Path) -> None:
        """Monitor loop processes an active session."""
        config = AcpConfig(
            polling_interval=1,
            tmux_session="test",
            compaction_enabled=True,
            memory_filing_enabled=True,
        )

        mock_session = MagicMock()
        mock_session.file_path = Path("/tmp/test.jsonl")

        mock_usage = MagicMock()
        mock_usage.total_context = 80000

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor") as MockTokenMonitor,
            patch("src.cli.DeliverySystem") as MockDelivery,
            patch("src.cli.CompactionTrigger") as MockCompaction,
            patch("src.cli.MemoryFilingTrigger") as MockMemory,
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = mock_session
            tracker.check_rotation.return_value = False
            tracker.get_read_position.return_value = 0

            token_mon = MockTokenMonitor.return_value
            token_mon.read_latest_usage.return_value = mock_usage
            token_mon.get_new_position.return_value = 100

            compaction = MockCompaction.return_value
            compaction.should_fire.return_value = True
            compaction.format_reminder.return_value = "[ACP] Compaction reminder"

            delivery = MockDelivery.return_value
            delivery.is_idle.return_value = True
            delivery.can_deliver.return_value = True

            memory = MockMemory.return_value
            memory.scan_for_milestones.return_value = []
            memory.get_new_position.return_value = 0

            # Break after one full cycle
            call_count = 0

            def sleep_side_effect(seconds: float) -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    raise KeyboardInterrupt

            mock_sleep.side_effect = sleep_side_effect

            run_monitor(config)

            # Verify the monitor processed the session
            tracker.find_active_session.assert_called()
            token_mon.read_latest_usage.assert_called()
            compaction.should_fire.assert_called()
            delivery.deliver.assert_called_once_with("[ACP] Compaction reminder", "compaction")
            compaction.record_reminder_sent.assert_called_once()

    def test_session_rotation_resets_state(self, tmp_pid: Path) -> None:
        """Session rotation resets session start time and compaction state."""
        config = AcpConfig(
            polling_interval=1,
            tmux_session="test",
            compaction_enabled=False,
            memory_filing_enabled=False,
        )

        mock_session = MagicMock()
        mock_session.file_path = Path("/tmp/test.jsonl")

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor"),
            patch("src.cli.DeliverySystem"),
            patch("src.cli.CompactionTrigger") as MockCompaction,
            patch("src.cli.MemoryFilingTrigger"),
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = mock_session
            tracker.check_rotation.return_value = True  # Rotation detected

            compaction = MockCompaction.return_value

            call_count = 0

            def sleep_side_effect(seconds: float) -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    raise KeyboardInterrupt

            mock_sleep.side_effect = sleep_side_effect

            run_monitor(config)

            compaction.record_compaction_detected.assert_called()

    def test_memory_filing_trigger(self, tmp_pid: Path) -> None:
        """Monitor fires memory filing reminder when milestone detected."""
        config = AcpConfig(
            polling_interval=1,
            tmux_session="test",
            compaction_enabled=False,
            memory_filing_enabled=True,
        )

        mock_session = MagicMock()
        mock_session.file_path = Path("/tmp/test.jsonl")

        mock_decision = MagicMock()
        mock_decision.action = "fire"

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor"),
            patch("src.cli.DeliverySystem") as MockDelivery,
            patch("src.cli.CompactionTrigger"),
            patch("src.cli.MemoryFilingTrigger") as MockMemory,
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = mock_session
            tracker.check_rotation.return_value = False

            delivery = MockDelivery.return_value
            delivery.is_idle.return_value = True
            delivery.can_deliver.return_value = True

            memory = MockMemory.return_value
            memory.scan_for_milestones.return_value = [MagicMock()]
            memory.get_new_position.return_value = 100
            memory.evaluate.return_value = mock_decision
            memory.format_reminder.return_value = "[ACP] Memory filing reminder"

            call_count = 0

            def sleep_side_effect(seconds: float) -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    raise KeyboardInterrupt

            mock_sleep.side_effect = sleep_side_effect

            run_monitor(config)

            delivery.deliver.assert_called_once_with(
                "[ACP] Memory filing reminder", "memory_filing"
            )
            memory.record_reminder_sent.assert_called_once()

    def test_signal_handler_stops_loop(self, tmp_pid: Path) -> None:
        """SIGTERM handler sets shutdown flag to exit the loop."""
        config = AcpConfig(polling_interval=1, tmux_session="test")

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor"),
            patch("src.cli.DeliverySystem"),
            patch("src.cli.CompactionTrigger"),
            patch("src.cli.MemoryFilingTrigger"),
            patch("src.cli.time.sleep") as mock_sleep,
            patch("src.cli.signal.signal") as mock_signal,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = None

            # Capture the signal handler
            handlers = {}

            def capture_handler(signum: int, handler: object) -> None:
                handlers[signum] = handler

            mock_signal.side_effect = capture_handler

            # Make sleep call the SIGTERM handler to simulate signal
            def sleep_side_effect(seconds: float) -> None:
                if signal.SIGTERM in handlers:
                    handlers[signal.SIGTERM](signal.SIGTERM, None)

            mock_sleep.side_effect = sleep_side_effect

            run_monitor(config)

            # PID file should be cleaned up
            assert not tmp_pid.is_file()

    def test_pid_file_cleanup_on_exit(self, tmp_pid: Path) -> None:
        """PID file is removed when the monitor exits."""
        config = AcpConfig(polling_interval=1, tmux_session="test")

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor"),
            patch("src.cli.DeliverySystem"),
            patch("src.cli.CompactionTrigger"),
            patch("src.cli.MemoryFilingTrigger"),
            patch("src.cli.time.sleep", side_effect=KeyboardInterrupt),
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = None

            run_monitor(config)

            assert not tmp_pid.is_file()

    def test_compaction_skipped_when_disabled(self, tmp_pid: Path) -> None:
        """When compaction_enabled is False, token monitor is not consulted."""
        config = AcpConfig(
            polling_interval=1,
            tmux_session="test",
            compaction_enabled=False,
            memory_filing_enabled=False,
        )

        mock_session = MagicMock()
        mock_session.file_path = Path("/tmp/test.jsonl")

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor") as MockTokenMonitor,
            patch("src.cli.DeliverySystem"),
            patch("src.cli.CompactionTrigger"),
            patch("src.cli.MemoryFilingTrigger"),
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = mock_session
            tracker.check_rotation.return_value = False

            token_mon = MockTokenMonitor.return_value

            mock_sleep.side_effect = KeyboardInterrupt

            run_monitor(config)

            token_mon.read_latest_usage.assert_not_called()

    def test_memory_filing_skipped_when_disabled(self, tmp_pid: Path) -> None:
        """When memory_filing_enabled is False, memory trigger is not consulted."""
        config = AcpConfig(
            polling_interval=1,
            tmux_session="test",
            compaction_enabled=False,
            memory_filing_enabled=False,
        )

        mock_session = MagicMock()
        mock_session.file_path = Path("/tmp/test.jsonl")

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor"),
            patch("src.cli.DeliverySystem"),
            patch("src.cli.CompactionTrigger"),
            patch("src.cli.MemoryFilingTrigger") as MockMemory,
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = mock_session
            tracker.check_rotation.return_value = False

            memory = MockMemory.return_value

            mock_sleep.side_effect = KeyboardInterrupt

            run_monitor(config)

            memory.scan_for_milestones.assert_not_called()

    def test_no_delivery_when_not_idle(self, tmp_pid: Path) -> None:
        """Compaction reminder is suppressed when delivery.is_idle returns False."""
        config = AcpConfig(
            polling_interval=1,
            tmux_session="test",
            compaction_enabled=True,
            memory_filing_enabled=False,
        )

        mock_session = MagicMock()
        mock_session.file_path = Path("/tmp/test.jsonl")

        mock_usage = MagicMock()
        mock_usage.total_context = 80000

        with (
            patch("src.cli.FileTracker") as MockTracker,
            patch("src.cli.TokenMonitor") as MockTokenMonitor,
            patch("src.cli.DeliverySystem") as MockDelivery,
            patch("src.cli.CompactionTrigger") as MockCompaction,
            patch("src.cli.MemoryFilingTrigger"),
            patch("src.cli.time.sleep") as mock_sleep,
        ):
            tracker = MockTracker.return_value
            tracker.find_active_session.return_value = mock_session
            tracker.check_rotation.return_value = False
            tracker.get_read_position.return_value = 0

            token_mon = MockTokenMonitor.return_value
            token_mon.read_latest_usage.return_value = mock_usage
            token_mon.get_new_position.return_value = 100

            compaction = MockCompaction.return_value
            compaction.should_fire.return_value = True

            delivery = MockDelivery.return_value
            delivery.is_idle.return_value = False  # Not idle

            mock_sleep.side_effect = KeyboardInterrupt

            run_monitor(config)

            delivery.deliver.assert_not_called()


# ---------------------------------------------------------------------------
# main() entry point tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No command shows help and returns 1."""
        result = main([])
        assert result == 1

    def test_no_command_prints_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No command shows usage text in stdout."""
        main([])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "acp" in captured.out.lower()

    def test_unknown_command(self) -> None:
        """Unknown command triggers argparse error."""
        with pytest.raises(SystemExit):
            main(["unknown"])

    def test_main_routes_to_init(self, tmp_path: Path) -> None:
        """main routes 'init' to cmd_init."""
        config_path = tmp_path / "test_config.yaml"
        result = main(["--config", str(config_path), "init"])
        assert result == 0
        assert config_path.is_file()

    def test_main_routes_to_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main routes 'config' to cmd_config."""
        result = main(["config"])
        assert result == 0
        captured = capsys.readouterr()
        assert "token_threshold" in captured.out

    def test_main_routes_to_status(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main routes 'status' to cmd_status."""
        result = main(["status"])
        assert result == 0

    def test_main_routes_to_stop(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main routes 'stop' to cmd_stop."""
        result = main(["stop"])
        assert result == 1  # No PID file, so returns 1
        captured = capsys.readouterr()
        assert "no pid file" in captured.out.lower()

    def test_main_routes_to_start_already_running(
        self, tmp_pid: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main routes 'start' to cmd_start (already running case)."""
        _write_pid(os.getpid())
        result = main(["start"])
        assert result == 1
        captured = capsys.readouterr()
        assert "already running" in captured.out.lower()

    def test_main_with_short_config_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main accepts -c as shorthand for --config."""
        result = main(["-c", "/nonexistent/path.yaml", "config"])
        assert result == 0
        captured = capsys.readouterr()
        assert "token_threshold" in captured.out
