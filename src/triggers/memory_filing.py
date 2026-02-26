"""Memory Filing Reminder Trigger for the Active Context Protocol.

Scans Claude Code session JSONL files for milestone events (PR merges, issue
closes, commits, PR creates) and reminds the agent to file learnings to
institutional memory.

The trigger detects milestones by scanning tool results and tool calls in the
JSONL stream, then applies a grace period (allowing the agent to file memory
unprompted) and a cooldown (preventing reminder spam).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .compaction import TriggerDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default milestone detection patterns
# ---------------------------------------------------------------------------

# Tool result patterns (content strings inside "user" type entries)
_RESULT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "pr_merge": [
        re.compile(r"Merged", re.IGNORECASE),
        re.compile(r"Pull request .* merged", re.IGNORECASE),
        re.compile(r"gh pr merge", re.IGNORECASE),
    ],
    "issue_close": [
        re.compile(r"gh issue close", re.IGNORECASE),
        re.compile(r"Closed issue", re.IGNORECASE),
        re.compile(r"Issue .* closed", re.IGNORECASE),
    ],
    "commit": [
        re.compile(r"^\[[\w/.-]+ [0-9a-f]{7,}\]", re.MULTILINE),
        re.compile(r"create mode \d+", re.IGNORECASE),
    ],
    "pr_create": [
        re.compile(r"gh pr create", re.IGNORECASE),
        re.compile(r"https://github\.com/.+/pull/\d+", re.IGNORECASE),
    ],
}

# Tool call patterns (input strings inside "assistant" type entries)
_CALL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "commit": [
        re.compile(r"git commit\b"),
    ],
    "pr_merge": [
        re.compile(r"gh pr merge\b"),
    ],
    "issue_close": [
        re.compile(r"gh issue close\b"),
    ],
    "pr_create": [
        re.compile(r"gh pr create\b"),
    ],
}

# Patterns that indicate the agent is already filing to memory
_MEMORY_FILING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"memory new\b"),
    re.compile(r"memory update\b"),
]


# ---------------------------------------------------------------------------
# MilestoneEvent dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MilestoneEvent:
    """A detected milestone event from the JSONL session stream.

    Attributes:
        event_type: One of ``"pr_merge"``, ``"issue_close"``, ``"commit"``,
            ``"pr_create"``.
        timestamp: ISO timestamp from the JSONL entry, or empty string if
            not available.
        details: Brief description extracted from the JSONL content.
    """

    event_type: str
    timestamp: str
    details: str


# ---------------------------------------------------------------------------
# MemoryFilingTrigger
# ---------------------------------------------------------------------------


class MemoryFilingTrigger:
    """Decides when to send a memory filing reminder to the agent.

    The trigger scans JSONL session files for milestone events (PR merges,
    issue closes, commits, PR creates). After detecting a milestone, it
    waits a grace period for the agent to file memory unprompted. If no
    memory filing is detected within the grace window, the trigger fires
    a reminder.

    Args:
        grace_after_event_seconds: Seconds to wait after a milestone
            before firing a reminder. If the agent files memory within
            this window, the reminder is suppressed.
        cooldown_seconds: Minimum seconds between consecutive reminders.
        patterns: Optional list of additional regex pattern strings to
            match in tool results as custom milestone indicators. Each
            match produces a ``"commit"`` type event (generic milestone).
    """

    def __init__(
        self,
        grace_after_event_seconds: int = 60,
        cooldown_seconds: int = 120,
        patterns: list[str] | None = None,
    ) -> None:
        self.grace_after_event_seconds = grace_after_event_seconds
        self.cooldown_seconds = cooldown_seconds

        self._custom_patterns: list[re.Pattern[str]] = []
        if patterns:
            for p in patterns:
                self._custom_patterns.append(re.compile(p))

        self._last_reminder_time: float | None = None
        self._pending: bool = False
        self._last_milestone_time: float | None = None
        self._last_position: int = 0
        self._memory_filed_since_milestone: bool = False

    # ------------------------------------------------------------------
    # JSONL scanning
    # ------------------------------------------------------------------

    def scan_for_milestones(
        self, jsonl_path: Path, from_position: int = 0
    ) -> list[MilestoneEvent]:
        """Scan a JSONL session file for milestone events.

        Reads from *from_position* to EOF, parsing each line for tool
        results and tool calls that indicate significant work milestones.

        Also detects memory filing calls (``memory new``, ``memory update``)
        and records them internally to support grace period suppression.

        Args:
            jsonl_path: Path to the Claude Code session JSONL file.
            from_position: Byte offset to start reading from.

        Returns:
            List of :class:`MilestoneEvent` instances found in the scanned
            region. Empty list if no milestones detected, the file does not
            exist, or the file is empty.
        """
        if not jsonl_path.is_file():
            logger.debug("JSONL file does not exist: %s", jsonl_path)
            self._last_position = from_position
            return []

        milestones: list[MilestoneEvent] = []

        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                f.seek(from_position)
                while True:
                    line = f.readline()
                    if not line:
                        self._last_position = f.tell()
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSONL line: %.80s", line
                        )
                        continue

                    if not isinstance(entry, dict):
                        continue

                    timestamp = entry.get("timestamp", "")
                    if not isinstance(timestamp, str):
                        timestamp = str(timestamp)

                    entry_type = entry.get("type", "")

                    # Scan tool results in "user" entries
                    if entry_type == "user":
                        milestones.extend(
                            self._scan_tool_results(entry, timestamp)
                        )
                        self._check_memory_filing_in_results(entry)

                    # Scan tool calls in "assistant" entries
                    elif entry_type == "assistant":
                        milestones.extend(
                            self._scan_tool_calls(entry, timestamp)
                        )
                        self._check_memory_filing_in_calls(entry)

        except OSError as exc:
            logger.warning("Error reading JSONL file %s: %s", jsonl_path, exc)
            self._last_position = from_position
            return []

        # Record milestone time if any found
        if milestones:
            self._last_milestone_time = time.time()

        return milestones

    def _scan_tool_results(
        self, entry: dict, timestamp: str
    ) -> list[MilestoneEvent]:
        """Extract milestone events from tool result content."""
        events: list[MilestoneEvent] = []
        message = entry.get("message")
        if not isinstance(message, dict):
            return events

        content = message.get("content")
        if not isinstance(content, list):
            return events

        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_result":
                continue

            text = item.get("content", "")
            if not isinstance(text, str):
                continue

            # Check default result patterns
            for event_type, patterns in _RESULT_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(text):
                        detail = _extract_detail(text, 120)
                        events.append(
                            MilestoneEvent(
                                event_type=event_type,
                                timestamp=timestamp,
                                details=detail,
                            )
                        )
                        break  # one match per event_type per item

            # Check custom patterns
            for pattern in self._custom_patterns:
                if pattern.search(text):
                    detail = _extract_detail(text, 120)
                    events.append(
                        MilestoneEvent(
                            event_type="custom",
                            timestamp=timestamp,
                            details=detail,
                        )
                    )
                    break

        return events

    def _scan_tool_calls(
        self, entry: dict, timestamp: str
    ) -> list[MilestoneEvent]:
        """Extract milestone events from tool call inputs."""
        events: list[MilestoneEvent] = []
        message = entry.get("message")
        if not isinstance(message, dict):
            return events

        content = message.get("content")
        if not isinstance(content, list):
            return events

        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_use":
                continue

            input_data = item.get("input", {})
            if not isinstance(input_data, dict):
                continue

            # Build a single text from input values to scan
            text = " ".join(
                str(v) for v in input_data.values() if isinstance(v, str)
            )

            for event_type, patterns in _CALL_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(text):
                        detail = _extract_detail(text, 120)
                        events.append(
                            MilestoneEvent(
                                event_type=event_type,
                                timestamp=timestamp,
                                details=detail,
                            )
                        )
                        break

            # Check custom patterns against tool call inputs
            for pattern in self._custom_patterns:
                if pattern.search(text):
                    detail = _extract_detail(text, 120)
                    events.append(
                        MilestoneEvent(
                            event_type="custom",
                            timestamp=timestamp,
                            details=detail,
                        )
                    )
                    break

        return events

    def _check_memory_filing_in_results(self, entry: dict) -> None:
        """Check tool results for memory filing commands."""
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return

        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("content", "")
            if not isinstance(text, str):
                continue
            for pattern in _MEMORY_FILING_PATTERNS:
                if pattern.search(text):
                    self._memory_filed_since_milestone = True
                    return

    def _check_memory_filing_in_calls(self, entry: dict) -> None:
        """Check tool calls for memory filing commands."""
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return

        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_use":
                continue
            input_data = item.get("input", {})
            if not isinstance(input_data, dict):
                continue
            text = " ".join(
                str(v) for v in input_data.values() if isinstance(v, str)
            )
            for pattern in _MEMORY_FILING_PATTERNS:
                if pattern.search(text):
                    self._memory_filed_since_milestone = True
                    return

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def evaluate(self, milestone_detected: bool) -> TriggerDecision:
        """Decide whether to fire a memory filing reminder.

        Checks are applied in order:
        1. No milestone detected -> skip
        2. Memory already filed since milestone -> skip
        3. Grace period active (milestone too recent) -> skip
        4. Cooldown active -> skip
        5. Pending reminder active -> skip
        6. Otherwise -> fire

        Args:
            milestone_detected: Whether any milestone events were found
                in the latest scan.

        Returns:
            A :class:`TriggerDecision` describing the outcome. The
            ``current_tokens`` and ``threshold`` fields are set to 0
            as they are not applicable to memory filing triggers.
        """
        now = time.time()

        # Check 1: No milestone
        if not milestone_detected and self._last_milestone_time is None:
            return TriggerDecision(
                action="skip_no_milestone",
                reason="No milestone events detected.",
                current_tokens=0,
                threshold=0,
            )

        # Check 2: Memory already filed
        if self._memory_filed_since_milestone:
            return TriggerDecision(
                action="skip_memory_filed",
                reason="Agent has already filed to memory since the last milestone.",
                current_tokens=0,
                threshold=0,
            )

        # Check 3: Grace period
        if self._last_milestone_time is not None:
            elapsed_since_milestone = now - self._last_milestone_time
            if elapsed_since_milestone < self.grace_after_event_seconds:
                remaining = self.grace_after_event_seconds - elapsed_since_milestone
                return TriggerDecision(
                    action="skip_grace_period",
                    reason=(
                        f"Grace period active ({elapsed_since_milestone:.0f}s of "
                        f"{self.grace_after_event_seconds}s elapsed, "
                        f"{remaining:.0f}s remaining)."
                    ),
                    current_tokens=0,
                    threshold=0,
                )

        # Check 4: Cooldown
        if self._last_reminder_time is not None:
            elapsed = now - self._last_reminder_time
            if elapsed < self.cooldown_seconds:
                return TriggerDecision(
                    action="skip_cooldown",
                    reason=(
                        f"Cooldown active ({elapsed:.0f}s of "
                        f"{self.cooldown_seconds}s elapsed)."
                    ),
                    current_tokens=0,
                    threshold=0,
                )

        # Check 5: Pending
        if self._pending:
            return TriggerDecision(
                action="skip_pending",
                reason="Reminder already pending. Waiting for memory filing.",
                current_tokens=0,
                threshold=0,
            )

        # All checks passed -> fire
        return TriggerDecision(
            action="fire",
            reason="Milestone detected and grace period expired. Recommending memory filing.",
            current_tokens=0,
            threshold=0,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def record_reminder_sent(self) -> None:
        """Record that a memory filing reminder was delivered.

        Sets the cooldown timer and marks the reminder as pending.
        Call this after successfully delivering a reminder message.
        """
        self._last_reminder_time = time.time()
        self._pending = True

    def record_memory_filed(self) -> None:
        """Record that the agent filed to memory.

        Resets the pending state and suppresses any pending reminder.
        This is the complement of :meth:`record_reminder_sent` -- it
        indicates the agent acted (either prompted or unprompted).
        """
        self._memory_filed_since_milestone = True
        self._pending = False

    def format_reminder(self) -> str:
        """Generate the memory filing reminder message text.

        Returns:
            A human-readable reminder string suitable for delivery via tmux.
        """
        return (
            "[ACP] Significant work detected. Consider filing to memory: "
            "takeaways, decisions, facts, or proverbs."
        )

    def get_new_position(self) -> int:
        """Return the byte position after the last :meth:`scan_for_milestones` call.

        Use this value as the ``from_position`` argument on the next call
        to :meth:`scan_for_milestones` for incremental reads.

        Returns:
            Byte offset in the file after the last scan completed.
        """
        return self._last_position

    def reset_for_new_milestone(self) -> None:
        """Reset memory-filed state for a new milestone cycle.

        Call this when starting to track a new set of milestones,
        e.g. after the previous reminder was acknowledged.
        """
        self._memory_filed_since_milestone = False
        self._last_milestone_time = None
        self._pending = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_detail(text: str, max_length: int = 120) -> str:
    """Extract a brief detail string from tool output text.

    Takes the first non-empty line and truncates to *max_length*.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if len(stripped) > max_length:
                return stripped[:max_length] + "..."
            return stripped
    return text[:max_length].strip() if text else ""
