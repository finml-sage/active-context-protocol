"""Compaction Reminder Trigger for the Active Context Protocol.

Evaluates whether a compaction reminder should be sent to the agent based on
current context token usage, session age, and cooldown/stacking constraints.

This trigger produces *reminders*, not automatic ``/compact`` commands.
The agent retains full control over when to act.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerDecision:
    """Result of a compaction trigger evaluation.

    Attributes:
        action: One of ``"fire"``, ``"skip_below_threshold"``,
            ``"skip_grace_period"``, ``"skip_cooldown"``, ``"skip_pending"``.
        reason: Human-readable explanation of the decision.
        current_tokens: The token count that was evaluated.
        threshold: The threshold that was applied.
    """

    action: str
    reason: str
    current_tokens: int
    threshold: int


class CompactionTrigger:
    """Decides when to send a compaction reminder to the agent.

    The trigger applies four checks in order:

    1. **Threshold** -- fire only when ``total_context_tokens > threshold``.
    2. **Grace period** -- suppress during the first ``grace_period_seconds``
       of a session so the agent has time to orient.
    3. **Cooldown** -- after sending a reminder, wait at least
       ``cooldown_seconds`` before sending another.
    4. **Stacking prevention** -- if a reminder was sent and no compaction
       has been detected since, the reminder is "pending".  Do not fire
       again until either compaction is detected or the cooldown expires.

    Args:
        threshold: Token count above which reminders become eligible.
        cooldown_seconds: Minimum seconds between consecutive reminders.
        grace_period_seconds: Seconds after session start during which
            reminders are suppressed.
    """

    def __init__(
        self,
        threshold: int = 70_000,
        cooldown_seconds: int = 120,
        grace_period_seconds: int = 300,
    ) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.grace_period_seconds = grace_period_seconds

        self._last_reminder_time: float | None = None
        self._pending: bool = False

    def evaluate(
        self, total_context_tokens: int, session_start_time: float
    ) -> TriggerDecision:
        """Evaluate whether a compaction reminder should fire.

        Checks are applied in the following order (first match wins):
        1. Below threshold -> skip
        2. Inside grace period -> skip
        3. Pending reminder with active cooldown -> skip (stacking)
        4. Inside cooldown (non-pending) -> skip
        5. Otherwise -> fire

        Args:
            total_context_tokens: Current total context token usage.
            session_start_time: Epoch timestamp (``time.time()``) of when
                the session started.

        Returns:
            A :class:`TriggerDecision` describing the outcome.
        """
        now = time.time()

        # Check 1: Threshold
        if total_context_tokens <= self.threshold:
            return TriggerDecision(
                action="skip_below_threshold",
                reason=(
                    f"Token count {total_context_tokens} is at or below "
                    f"threshold {self.threshold}."
                ),
                current_tokens=total_context_tokens,
                threshold=self.threshold,
            )

        # Check 2: Grace period
        session_age = now - session_start_time
        if session_age < self.grace_period_seconds:
            remaining = self.grace_period_seconds - session_age
            return TriggerDecision(
                action="skip_grace_period",
                reason=(
                    f"Session is {session_age:.0f}s old; grace period "
                    f"({self.grace_period_seconds}s) has {remaining:.0f}s remaining."
                ),
                current_tokens=total_context_tokens,
                threshold=self.threshold,
            )

        # Check 3: Stacking prevention -- pending reminder + cooldown active
        if self._pending and self._last_reminder_time is not None:
            elapsed = now - self._last_reminder_time
            if elapsed < self.cooldown_seconds:
                return TriggerDecision(
                    action="skip_pending",
                    reason=(
                        f"Reminder already pending (sent {elapsed:.0f}s ago). "
                        f"Waiting for compaction or cooldown expiry "
                        f"({self.cooldown_seconds - elapsed:.0f}s remaining)."
                    ),
                    current_tokens=total_context_tokens,
                    threshold=self.threshold,
                )
            # Cooldown expired while pending -- allow re-fire
            self._pending = False

        # Check 4: Cooldown (non-pending)
        if self._last_reminder_time is not None:
            elapsed = now - self._last_reminder_time
            if elapsed < self.cooldown_seconds:
                return TriggerDecision(
                    action="skip_cooldown",
                    reason=(
                        f"Cooldown active ({elapsed:.0f}s of "
                        f"{self.cooldown_seconds}s elapsed)."
                    ),
                    current_tokens=total_context_tokens,
                    threshold=self.threshold,
                )

        # All checks passed -> fire
        return TriggerDecision(
            action="fire",
            reason=(
                f"Token count {total_context_tokens} exceeds threshold "
                f"{self.threshold}. Recommending compaction."
            ),
            current_tokens=total_context_tokens,
            threshold=self.threshold,
        )

    def record_reminder_sent(self) -> None:
        """Record that a compaction reminder was delivered.

        Sets the cooldown timer and marks the reminder as pending (for
        stacking prevention).  Call this after successfully delivering a
        reminder message.
        """
        self._last_reminder_time = time.time()
        self._pending = True

    def record_compaction_detected(self) -> None:
        """Record that compaction occurred (e.g. token count dropped sharply).

        Resets the pending state and clears the cooldown timer so the
        trigger is ready to fire again when the threshold is next exceeded.
        """
        self._pending = False
        self._last_reminder_time = None

    def should_fire(
        self, total_context_tokens: int, session_start_time: float
    ) -> bool:
        """Convenience method: evaluate and return True if action is ``"fire"``.

        Args:
            total_context_tokens: Current total context token usage.
            session_start_time: Epoch timestamp of session start.

        Returns:
            True if a reminder should be sent, False otherwise.
        """
        decision = self.evaluate(total_context_tokens, session_start_time)
        return decision.action == "fire"

    def format_reminder(self, total_context_tokens: int) -> str:
        """Generate the reminder message text.

        Args:
            total_context_tokens: Current total context token usage.

        Returns:
            A human-readable reminder string suitable for delivery via tmux.
        """
        pct = int(round(total_context_tokens / self.threshold * 100))
        return (
            f"[ACP] Context at {total_context_tokens} tokens "
            f"({pct}% of {self.threshold} threshold). "
            f"Consider compacting when ready."
        )
