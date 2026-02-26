"""CLI entry point for the Active Context Protocol.

Provides a simple ``acp`` command with subcommands for starting, stopping,
and inspecting the monitor.  Uses only ``argparse`` (no Click/Typer dependency).

Commands::

    acp start   — Start the monitor (enters main loop)
    acp stop    — Stop a running monitor (sends SIGTERM)
    acp status  — Show current status (PID, running?, config path)
    acp config  — Show resolved configuration
    acp init    — Create default config file at ~/.acp/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import fields
from pathlib import Path

from .config import AcpConfig, _DEFAULT_CONFIG_PATH, load_config, save_default_config
from .delivery import DeliverySystem
from .file_tracker import FileTracker
from .token_monitor import TokenMonitor
from .triggers.compaction import CompactionTrigger
from .triggers.memory_filing import MemoryFilingTrigger

logger = logging.getLogger(__name__)

_PID_FILE = Path("~/.acp/acp.pid")


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


def _write_pid(pid: int | None = None) -> Path:
    """Write the current PID to the PID file.  Returns the resolved path."""
    pid_path = _PID_FILE.expanduser()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid or os.getpid()), encoding="utf-8")
    return pid_path


def _read_pid() -> int | None:
    """Read the PID from the PID file, or None if missing/invalid."""
    pid_path = _PID_FILE.expanduser()
    if not pid_path.is_file():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _remove_pid() -> None:
    """Remove the PID file if it exists."""
    pid_path = _PID_FILE.expanduser()
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _is_process_running(pid: int) -> bool:
    """Check whether a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------


def run_monitor(config: AcpConfig) -> None:
    """Run the ACP monitoring loop.

    Sets up all components from the config, writes a PID file, installs
    signal handlers for clean shutdown, and enters the polling loop.

    Args:
        config: Resolved ACP configuration.
    """
    tracker = FileTracker()
    token_monitor = TokenMonitor(threshold=config.token_threshold)
    delivery = DeliverySystem(
        tmux_session=config.tmux_session,
        warmdown_seconds=config.warmdown_interval,
        idle_threshold_seconds=config.idle_threshold,
        log_file=Path(config.log_file).expanduser(),
    )
    compaction_trigger = CompactionTrigger(
        threshold=config.compaction_threshold,
        cooldown_seconds=config.compaction_cooldown,
        grace_period_seconds=config.grace_period,
    )
    memory_trigger = MemoryFilingTrigger(
        grace_after_event_seconds=config.memory_filing_grace_after_event,
        cooldown_seconds=config.warmdown_interval,
        patterns=config.memory_filing_patterns or None,
    )

    # PID file
    pid_path = _write_pid()
    logger.info("ACP monitor started (PID %d, pid file %s)", os.getpid(), pid_path)

    # Signal handling for clean shutdown
    shutdown_requested = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Received signal %d — shutting down", signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    session_start_time = time.time()

    try:
        while not shutdown_requested:
            session = tracker.find_active_session()
            if session is None:
                time.sleep(config.polling_interval)
                continue

            if tracker.check_rotation():
                session_start_time = time.time()
                compaction_trigger.record_compaction_detected()  # Reset on new session

            # Check token usage
            if config.compaction_enabled:
                usage = token_monitor.read_latest_usage(
                    session.file_path, tracker.get_read_position()
                )
                tracker.update_read_position(token_monitor.get_new_position())

                if usage and compaction_trigger.should_fire(
                    usage.total_context, session_start_time
                ):
                    if delivery.is_idle(session.file_path) and delivery.can_deliver():
                        msg = compaction_trigger.format_reminder(usage.total_context)
                        delivery.deliver(msg, "compaction")
                        compaction_trigger.record_reminder_sent()

            # Check memory filing
            if config.memory_filing_enabled:
                milestones = memory_trigger.scan_for_milestones(
                    session.file_path, memory_trigger.get_new_position()
                )
                if milestones:
                    decision = memory_trigger.evaluate(milestone_detected=True)
                    if decision.action == "fire":
                        if delivery.is_idle(
                            session.file_path
                        ) and delivery.can_deliver():
                            msg = memory_trigger.format_reminder()
                            delivery.deliver(msg, "memory_filing")
                            memory_trigger.record_reminder_sent()

            time.sleep(config.polling_interval)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        _remove_pid()
        logger.info("ACP monitor stopped")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> int:
    """Handle ``acp start``."""
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Check if already running
    existing_pid = _read_pid()
    if existing_pid is not None and _is_process_running(existing_pid):
        print(f"ACP monitor already running (PID {existing_pid})")
        return 1

    print("Starting ACP monitor...")
    if config.tmux_session:
        print(f"  tmux session: {config.tmux_session}")
    else:
        print("  WARNING: no tmux_session configured — delivery will be skipped")
    print(f"  polling interval: {config.polling_interval}s")
    print(f"  token threshold: {config.token_threshold}")

    run_monitor(config)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Handle ``acp stop``."""
    pid = _read_pid()
    if pid is None:
        print("No PID file found — ACP monitor may not be running")
        return 1

    if not _is_process_running(pid):
        print(f"Process {pid} is not running — cleaning up stale PID file")
        _remove_pid()
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to ACP monitor (PID {pid})")
        _remove_pid()
        return 0
    except OSError as exc:
        print(f"Failed to stop ACP monitor (PID {pid}): {exc}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Handle ``acp status``."""
    config_path = (Path(args.config) if args.config else _DEFAULT_CONFIG_PATH).expanduser()

    pid = _read_pid()
    if pid is not None and _is_process_running(pid):
        print(f"ACP monitor is running (PID {pid})")
    elif pid is not None:
        print(f"ACP monitor is NOT running (stale PID {pid})")
    else:
        print("ACP monitor is not running")

    print(f"Config: {config_path}")
    print(f"PID file: {_PID_FILE.expanduser()}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Handle ``acp config``."""
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    print("Resolved ACP configuration:")
    print("-" * 40)
    for f in fields(config):
        value = getattr(config, f.name)
        print(f"  {f.name}: {value!r}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Handle ``acp init``."""
    config_path = (Path(args.config) if args.config else _DEFAULT_CONFIG_PATH).expanduser()

    if config_path.is_file() and not args.force:
        print(f"Config already exists at {config_path}")
        print("Use --force to overwrite")
        return 1

    save_default_config(config_path)
    print(f"Default config written to {config_path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="acp",
        description="Active Context Protocol — context management for Claude Code agents",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config file (default: ~/.acp/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    subparsers.add_parser("start", help="Start the ACP monitor")

    # stop
    subparsers.add_parser("stop", help="Stop a running ACP monitor")

    # status
    subparsers.add_parser("status", help="Show ACP monitor status")

    # config
    subparsers.add_parser("config", help="Show resolved configuration")

    # init
    init_parser = subparsers.add_parser("init", help="Create default config file")
    init_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing config"
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "config": cmd_config,
        "init": cmd_init,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
