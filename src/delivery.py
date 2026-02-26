"""tmux Delivery System for the Active Context Protocol.

Delivers reminder messages to an agent's tmux session with idle detection,
collision avoidance, and global warmdown enforcement.

Known failure modes this module guards against:
- Firing during active tool processing corrupts the conversation
- User typing collision concatenates messages
- tmux send-keys requires two separate subprocess calls (text, then Enter)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Valid delivery modes
DeliveryMode = Literal["reminder", "command"]

# Valid outcomes for delivery attempts
DeliveryOutcome = Literal[
    "delivered",
    "queued_not_idle",
    "queued_warmdown",
    "skipped_no_tmux",
    "skipped_no_session",
    "failed",
]


@dataclass
class DeliveryResult:
    """Result of a delivery attempt."""

    success: bool
    outcome: DeliveryOutcome
    timestamp: datetime
    trigger_type: str
    message: str


class DeliverySystem:
    """Delivers reminder messages to an agent's tmux session.

    Implements idle detection, global warmdown, and the proven two-call
    tmux send-keys pattern. All delivery attempts are logged to an audit
    trail regardless of outcome.

    Args:
        tmux_session: Name of the target tmux session.
        warmdown_seconds: Minimum seconds between deliveries. Default 120.
        idle_threshold_seconds: Seconds since last JSONL write to consider
            the agent idle. Default 5.
        log_file: Path for the audit trail. Falls back to stderr if None.
    """

    def __init__(
        self,
        tmux_session: str,
        warmdown_seconds: int = 120,
        idle_threshold_seconds: float = 5.0,
        log_file: Path | None = None,
    ) -> None:
        self.tmux_session = tmux_session
        self.warmdown_seconds = warmdown_seconds
        self.idle_threshold_seconds = idle_threshold_seconds
        self.log_file = log_file
        self._last_delivery_time: datetime | None = None

        # Configure audit trail logging — unique logger per instance to avoid
        # cross-contamination when multiple DeliverySystem instances exist
        # (e.g., in tests with different log file paths).
        self._audit_logger = logging.getLogger(
            f"{__name__}.audit.{id(self)}"
        )
        self._audit_logger.setLevel(logging.INFO)
        self._audit_logger.propagate = False
        # Clear any pre-existing handlers (defensive)
        self._audit_logger.handlers.clear()

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.FileHandler(
                str(log_file), mode="a"
            )
        else:
            handler = logging.StreamHandler(sys.stderr)

        handler.setFormatter(logging.Formatter("%(message)s"))
        self._audit_logger.addHandler(handler)

    def _audit(
        self,
        trigger_type: str,
        outcome: DeliveryOutcome,
        message: str,
        mode: DeliveryMode = "reminder",
    ) -> None:
        """Append an entry to the audit trail."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        preview = message[:80].replace("\n", " ")
        self._audit_logger.info(
            f"{now} | {trigger_type} | {outcome} | mode={mode} | {preview}"
        )

    def is_idle(self, jsonl_path: Path) -> bool:
        """Check whether the agent is idle based on its JSONL session log.

        Both conditions must be true for idle:
        1. The JSONL file's mtime is more than ``idle_threshold_seconds`` ago.
        2. The last entry in the JSONL is an ``assistant`` type (not a
           ``progress`` or ``tool_use`` in-flight entry).

        If the file cannot be read or parsed, this method returns False
        (erring on the side of NOT delivering).

        Args:
            jsonl_path: Path to the Claude Code session JSONL file.

        Returns:
            True if the agent appears idle and safe to deliver to.
        """
        try:
            if not jsonl_path.exists():
                logger.debug("JSONL path does not exist: %s", jsonl_path)
                return False

            # Condition 1: mtime staleness
            mtime = jsonl_path.stat().st_mtime
            age = time.time() - mtime
            if age < self.idle_threshold_seconds:
                logger.debug(
                    "JSONL too recent (%.1fs < %.1fs threshold)",
                    age,
                    self.idle_threshold_seconds,
                )
                return False

            # Condition 2: last entry type
            last_line = self._read_last_line(jsonl_path)
            if last_line is None:
                logger.debug("Could not read last line of JSONL")
                return False

            try:
                entry = json.loads(last_line)
            except json.JSONDecodeError:
                logger.debug("Failed to parse last JSONL line as JSON")
                return False

            entry_type = entry.get("type", "")
            if entry_type != "assistant":
                logger.debug(
                    "Last JSONL entry type is '%s', not 'assistant'", entry_type
                )
                return False

            # Check for tool_use blocks in assistant content — if present,
            # a tool call was just initiated and the agent is busy, not idle.
            content = entry.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        logger.debug(
                            "Last assistant entry contains tool_use block — agent is busy"
                        )
                        return False

            return True

        except OSError as exc:
            logger.warning("Error checking idle state: %s", exc)
            return False

    def can_deliver(self) -> bool:
        """Check whether the global warmdown period has elapsed.

        Returns:
            True if enough time has passed since the last successful delivery.
        """
        if self._last_delivery_time is None:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self._last_delivery_time
        ).total_seconds()
        return elapsed >= self.warmdown_seconds

    def deliver(
        self,
        message: str,
        trigger_type: str,
        mode: DeliveryMode = "reminder",
        jsonl_path: Path | None = None,
    ) -> DeliveryResult:
        """Attempt to deliver a message to the tmux session.

        Checks preconditions in order:
        1. tmux binary available
        2. Target session exists
        3. In ``command`` mode: agent must be idle (requires ``jsonl_path``)
        4. Global warmdown elapsed

        In ``reminder`` mode (the default), idle detection is skipped.
        Reminders are text input that can arrive any time — the agent
        processes them on its next turn.  Only the warmdown gate applies.

        In ``command`` mode, idle detection is enforced because commands
        (like ``/compact``) must arrive at the idle input prompt to be
        parsed as CLI slash commands rather than user text.

        Args:
            message: The text to deliver to the tmux session.
            trigger_type: Category of the trigger (e.g., "compaction",
                "memory_filing", "custom").
            mode: Delivery mode — ``"reminder"`` (default) skips idle
                detection; ``"command"`` requires idle.
            jsonl_path: Path to the JSONL session file.  Required when
                ``mode="command"`` so ``is_idle()`` can be evaluated.

        Returns:
            A DeliveryResult describing what happened.
        """
        now = datetime.now(timezone.utc)

        # Check 1: tmux binary
        if not self._tmux_available():
            result = DeliveryResult(
                success=False,
                outcome="skipped_no_tmux",
                timestamp=now,
                trigger_type=trigger_type,
                message=message,
            )
            self._audit(trigger_type, result.outcome, message, mode)
            return result

        # Check 2: session exists
        if not self._session_exists():
            result = DeliveryResult(
                success=False,
                outcome="skipped_no_session",
                timestamp=now,
                trigger_type=trigger_type,
                message=message,
            )
            self._audit(trigger_type, result.outcome, message, mode)
            return result

        # Check 3: idle detection (command mode only)
        if mode == "command":
            if jsonl_path is None:
                logger.error(
                    "command mode delivery requires jsonl_path, got None"
                )
                result = DeliveryResult(
                    success=False,
                    outcome="failed",
                    timestamp=now,
                    trigger_type=trigger_type,
                    message=message,
                )
                self._audit(trigger_type, result.outcome, message, mode)
                return result
            if not self.is_idle(jsonl_path):
                result = DeliveryResult(
                    success=False,
                    outcome="queued_not_idle",
                    timestamp=now,
                    trigger_type=trigger_type,
                    message=message,
                )
                self._audit(trigger_type, result.outcome, message, mode)
                return result

        # Check 4: warmdown
        if not self.can_deliver():
            result = DeliveryResult(
                success=False,
                outcome="queued_warmdown",
                timestamp=now,
                trigger_type=trigger_type,
                message=message,
            )
            self._audit(trigger_type, result.outcome, message, mode)
            return result

        # All preconditions met — deliver
        try:
            self._send_to_tmux(message)
            self._last_delivery_time = datetime.now(timezone.utc)
            result = DeliveryResult(
                success=True,
                outcome="delivered",
                timestamp=now,
                trigger_type=trigger_type,
                message=message,
            )
            self._audit(trigger_type, result.outcome, message, mode)
            return result
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.error("tmux delivery failed: %s", exc)
            result = DeliveryResult(
                success=False,
                outcome="failed",
                timestamp=now,
                trigger_type=trigger_type,
                message=message,
            )
            self._audit(trigger_type, result.outcome, message, mode)
            return result

    # --- Private helpers ---

    @staticmethod
    def _tmux_available() -> bool:
        """Check if tmux is installed."""
        return shutil.which("tmux") is not None

    def _session_exists(self) -> bool:
        """Check if the target tmux session exists."""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.tmux_session],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("tmux session check failed: %s", exc)
            return False

    def _send_to_tmux(self, message: str) -> None:
        """Send message to tmux using the proven two-call pattern.

        Two separate subprocess calls are required because a single call
        with both message and Enter silently drops the Enter keystroke.
        """
        # Call 1: send the message text
        subprocess.run(
            ["tmux", "send-keys", "-t", self.tmux_session, message],
            check=True,
            capture_output=True,
            timeout=10,
        )
        # Brief pause between calls
        time.sleep(0.5)
        # Call 2: send Enter separately
        subprocess.run(
            ["tmux", "send-keys", "-t", self.tmux_session, "Enter"],
            check=True,
            capture_output=True,
            timeout=10,
        )

    @staticmethod
    def _read_last_line(path: Path) -> str | None:
        """Read the last non-empty line of a file efficiently.

        Uses seek-from-end to avoid reading the entire file into memory.
        Falls back to full read for small files.
        """
        try:
            size = path.stat().st_size
            if size == 0:
                return None

            with path.open("rb") as f:
                # For small files, just read everything
                if size <= 8192:
                    content = f.read().decode("utf-8", errors="replace")
                    lines = content.strip().splitlines()
                    return lines[-1] if lines else None

                # For large files, seek from end
                f.seek(0, 2)  # End of file
                pos = f.tell()
                buf = b""
                # Read backwards in chunks until we find a newline
                while pos > 0:
                    chunk_size = min(4096, pos)
                    pos -= chunk_size
                    f.seek(pos)
                    buf = f.read(chunk_size) + buf
                    lines = buf.strip().split(b"\n")
                    if len(lines) > 1:
                        return lines[-1].decode("utf-8", errors="replace")

                # File has only one line
                return buf.strip().decode("utf-8", errors="replace") or None

        except OSError as exc:
            logger.warning("Failed to read last line of %s: %s", path, exc)
            return None
