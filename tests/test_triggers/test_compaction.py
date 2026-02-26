"""Comprehensive tests for the Compaction Reminder Trigger."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.triggers.compaction import CompactionTrigger, TriggerDecision


# ---------------------------------------------------------------------------
# TriggerDecision dataclass
# ---------------------------------------------------------------------------

class TestTriggerDecision:
    """Tests for the TriggerDecision dataclass."""

    def test_fields(self) -> None:
        decision = TriggerDecision(
            action="fire",
            reason="Threshold exceeded.",
            current_tokens=80_000,
            threshold=70_000,
        )
        assert decision.action == "fire"
        assert decision.reason == "Threshold exceeded."
        assert decision.current_tokens == 80_000
        assert decision.threshold == 70_000

    def test_frozen(self) -> None:
        decision = TriggerDecision(
            action="fire",
            reason="test",
            current_tokens=0,
            threshold=0,
        )
        with pytest.raises(AttributeError):
            decision.action = "skip_cooldown"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CompactionTrigger — threshold checks
# ---------------------------------------------------------------------------

class TestThreshold:
    """Tests for the threshold gate."""

    def test_below_threshold_skips(self) -> None:
        """Token count below threshold -> skip_below_threshold."""
        trigger = CompactionTrigger(threshold=70_000)
        session_start = time.time() - 1000  # well past grace period
        decision = trigger.evaluate(50_000, session_start)

        assert decision.action == "skip_below_threshold"
        assert decision.current_tokens == 50_000
        assert decision.threshold == 70_000

    def test_at_threshold_skips(self) -> None:
        """Token count exactly at threshold -> skip (not strictly above)."""
        trigger = CompactionTrigger(threshold=70_000)
        session_start = time.time() - 1000
        decision = trigger.evaluate(70_000, session_start)

        assert decision.action == "skip_below_threshold"

    def test_above_threshold_fires(self) -> None:
        """Token count above threshold -> fire."""
        trigger = CompactionTrigger(threshold=70_000)
        session_start = time.time() - 1000
        decision = trigger.evaluate(70_001, session_start)

        assert decision.action == "fire"
        assert decision.current_tokens == 70_001
        assert decision.threshold == 70_000

    def test_zero_tokens_skips(self) -> None:
        """Zero tokens -> skip_below_threshold."""
        trigger = CompactionTrigger(threshold=70_000)
        session_start = time.time() - 1000
        decision = trigger.evaluate(0, session_start)

        assert decision.action == "skip_below_threshold"

    def test_very_large_tokens_fires(self) -> None:
        """Very large token count -> fire."""
        trigger = CompactionTrigger(threshold=70_000)
        session_start = time.time() - 1000
        decision = trigger.evaluate(10_000_000, session_start)

        assert decision.action == "fire"


# ---------------------------------------------------------------------------
# CompactionTrigger — grace period
# ---------------------------------------------------------------------------

class TestGracePeriod:
    """Tests for the grace period gate."""

    def test_within_grace_period_skips(self) -> None:
        """Inside grace period -> skip_grace_period, even above threshold."""
        trigger = CompactionTrigger(threshold=70_000, grace_period_seconds=600)
        session_start = time.time() - 100  # only 100s into session
        decision = trigger.evaluate(80_000, session_start)

        assert decision.action == "skip_grace_period"
        assert "grace period" in decision.reason.lower()

    def test_grace_period_just_expired_fires(self) -> None:
        """Just past grace period -> fire."""
        trigger = CompactionTrigger(threshold=70_000, grace_period_seconds=600)
        session_start = time.time() - 601  # 1s past grace
        decision = trigger.evaluate(80_000, session_start)

        assert decision.action == "fire"

    def test_grace_period_zero_allows_immediate(self) -> None:
        """Grace period of 0 -> fires immediately if above threshold."""
        trigger = CompactionTrigger(
            threshold=70_000, grace_period_seconds=0
        )
        session_start = time.time()
        decision = trigger.evaluate(80_000, session_start)

        assert decision.action == "fire"


# ---------------------------------------------------------------------------
# CompactionTrigger — cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    """Tests for the cooldown gate."""

    def test_cooldown_active_skips(self) -> None:
        """After a reminder, within cooldown -> skip_cooldown."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=120, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        # Fire once
        d1 = trigger.evaluate(80_000, session_start)
        assert d1.action == "fire"
        trigger.record_reminder_sent()

        # Then detect compaction to clear pending state
        trigger.record_compaction_detected()

        # Manually set last_reminder_time to recent to test pure cooldown
        trigger._last_reminder_time = time.time()
        trigger._pending = False

        decision = trigger.evaluate(80_000, session_start)
        assert decision.action == "skip_cooldown"

    def test_cooldown_expired_fires(self) -> None:
        """After cooldown expires -> fire again."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=120, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        # Simulate reminder sent long ago
        trigger._last_reminder_time = time.time() - 200  # 200s ago, cooldown is 120s
        trigger._pending = False

        decision = trigger.evaluate(80_000, session_start)
        assert decision.action == "fire"

    def test_cooldown_zero_fires_immediately(self) -> None:
        """Cooldown of 0 -> fires immediately after previous reminder."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=0, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        d1 = trigger.evaluate(80_000, session_start)
        assert d1.action == "fire"
        trigger.record_reminder_sent()
        trigger.record_compaction_detected()

        d2 = trigger.evaluate(80_000, session_start)
        assert d2.action == "fire"


# ---------------------------------------------------------------------------
# CompactionTrigger — stacking prevention
# ---------------------------------------------------------------------------

class TestStackingPrevention:
    """Tests for the stacking prevention gate."""

    def test_pending_reminder_skips(self) -> None:
        """Reminder sent, no compaction detected, cooldown active -> skip_pending."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=120, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        # Fire and record
        d1 = trigger.evaluate(80_000, session_start)
        assert d1.action == "fire"
        trigger.record_reminder_sent()

        # Try again immediately -> pending
        d2 = trigger.evaluate(80_000, session_start)
        assert d2.action == "skip_pending"

    def test_pending_clears_after_cooldown(self) -> None:
        """Pending reminder + cooldown expired -> fire again."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=120, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        # Fire and record
        trigger.evaluate(80_000, session_start)
        trigger.record_reminder_sent()

        # Simulate cooldown expiry
        trigger._last_reminder_time = time.time() - 200

        decision = trigger.evaluate(80_000, session_start)
        assert decision.action == "fire"
        # Pending should have been cleared
        assert trigger._pending is False

    def test_compaction_detected_resets_pending(self) -> None:
        """Compaction detected -> clears pending, allows re-fire."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=120, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        # Fire and record
        trigger.evaluate(80_000, session_start)
        trigger.record_reminder_sent()

        # Compaction detected
        trigger.record_compaction_detected()

        # Should be able to fire immediately (no cooldown, no pending)
        decision = trigger.evaluate(80_000, session_start)
        assert decision.action == "fire"


# ---------------------------------------------------------------------------
# CompactionTrigger — compaction detection
# ---------------------------------------------------------------------------

class TestCompactionDetected:
    """Tests for record_compaction_detected."""

    def test_resets_pending(self) -> None:
        trigger = CompactionTrigger(grace_period_seconds=0)
        trigger._pending = True
        trigger._last_reminder_time = time.time()

        trigger.record_compaction_detected()

        assert trigger._pending is False
        assert trigger._last_reminder_time is None

    def test_resets_cooldown_timer(self) -> None:
        trigger = CompactionTrigger(grace_period_seconds=0)
        trigger._last_reminder_time = time.time()

        trigger.record_compaction_detected()

        assert trigger._last_reminder_time is None

    def test_idempotent_when_clean(self) -> None:
        """Calling record_compaction_detected on a clean trigger is safe."""
        trigger = CompactionTrigger()
        trigger.record_compaction_detected()  # should not raise
        assert trigger._pending is False
        assert trigger._last_reminder_time is None


# ---------------------------------------------------------------------------
# CompactionTrigger — should_fire convenience method
# ---------------------------------------------------------------------------

class TestShouldFire:
    """Tests for the should_fire convenience method."""

    def test_returns_true_when_fire(self) -> None:
        trigger = CompactionTrigger(threshold=70_000, grace_period_seconds=0)
        session_start = time.time() - 1000

        assert trigger.should_fire(80_000, session_start) is True

    def test_returns_false_when_skip(self) -> None:
        trigger = CompactionTrigger(threshold=70_000)
        session_start = time.time() - 1000

        assert trigger.should_fire(50_000, session_start) is False

    def test_returns_false_during_grace(self) -> None:
        trigger = CompactionTrigger(threshold=70_000, grace_period_seconds=600)
        session_start = time.time() - 100

        assert trigger.should_fire(80_000, session_start) is False


# ---------------------------------------------------------------------------
# CompactionTrigger — format_reminder
# ---------------------------------------------------------------------------

class TestFormatReminder:
    """Tests for reminder message formatting."""

    def test_default_format(self) -> None:
        trigger = CompactionTrigger(threshold=70_000)
        msg = trigger.format_reminder(80_000)

        assert "[ACP]" in msg
        assert "80000" in msg
        assert "70000" in msg
        assert "114%" in msg  # 80000/70000 = ~114%
        assert "Consider compacting" in msg

    def test_at_threshold(self) -> None:
        trigger = CompactionTrigger(threshold=70_000)
        msg = trigger.format_reminder(70_000)

        assert "100%" in msg

    def test_double_threshold(self) -> None:
        trigger = CompactionTrigger(threshold=70_000)
        msg = trigger.format_reminder(140_000)

        assert "200%" in msg

    def test_custom_threshold(self) -> None:
        trigger = CompactionTrigger(threshold=100_000)
        msg = trigger.format_reminder(150_000)

        assert "150000" in msg
        assert "100000" in msg
        assert "150%" in msg

    def test_small_tokens(self) -> None:
        trigger = CompactionTrigger(threshold=70_000)
        msg = trigger.format_reminder(1000)

        assert "1000" in msg
        assert "1%" in msg  # 1000/70000 ~= 1.4%, rounds to 1


# ---------------------------------------------------------------------------
# CompactionTrigger — record_reminder_sent
# ---------------------------------------------------------------------------

class TestRecordReminderSent:
    """Tests for record_reminder_sent."""

    def test_sets_pending(self) -> None:
        trigger = CompactionTrigger()
        assert trigger._pending is False
        trigger.record_reminder_sent()
        assert trigger._pending is True

    def test_sets_last_reminder_time(self) -> None:
        trigger = CompactionTrigger()
        assert trigger._last_reminder_time is None
        before = time.time()
        trigger.record_reminder_sent()
        after = time.time()
        assert trigger._last_reminder_time is not None
        assert before <= trigger._last_reminder_time <= after


# ---------------------------------------------------------------------------
# CompactionTrigger — constructor defaults
# ---------------------------------------------------------------------------

class TestConstructorDefaults:
    """Tests for constructor parameter defaults."""

    def test_default_threshold(self) -> None:
        trigger = CompactionTrigger()
        assert trigger.threshold == 70_000

    def test_default_cooldown(self) -> None:
        trigger = CompactionTrigger()
        assert trigger.cooldown_seconds == 120

    def test_default_grace_period(self) -> None:
        trigger = CompactionTrigger()
        assert trigger.grace_period_seconds == 600

    def test_custom_values(self) -> None:
        trigger = CompactionTrigger(
            threshold=50_000,
            cooldown_seconds=60,
            grace_period_seconds=300,
        )
        assert trigger.threshold == 50_000
        assert trigger.cooldown_seconds == 60
        assert trigger.grace_period_seconds == 300

    def test_initial_state_clean(self) -> None:
        trigger = CompactionTrigger()
        assert trigger._last_reminder_time is None
        assert trigger._pending is False


# ---------------------------------------------------------------------------
# CompactionTrigger — full lifecycle integration
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """Integration-style tests combining multiple features."""

    def test_session_lifecycle(self) -> None:
        """Simulate a full session: start -> build up -> remind -> compact -> remind again."""
        trigger = CompactionTrigger(
            threshold=70_000,
            cooldown_seconds=120,
            grace_period_seconds=600,
        )
        session_start = time.time() - 700  # 700s ago, past grace

        # Phase 1: Below threshold
        d = trigger.evaluate(30_000, session_start)
        assert d.action == "skip_below_threshold"

        # Phase 2: Cross threshold -> fire
        d = trigger.evaluate(75_000, session_start)
        assert d.action == "fire"
        trigger.record_reminder_sent()

        # Phase 3: Try again immediately -> pending
        d = trigger.evaluate(80_000, session_start)
        assert d.action == "skip_pending"

        # Phase 4: Compaction happens (tokens drop)
        trigger.record_compaction_detected()

        # Phase 5: Tokens climb again -> fire
        d = trigger.evaluate(72_000, session_start)
        assert d.action == "fire"
        trigger.record_reminder_sent()

        # Phase 6: Pending again
        d = trigger.evaluate(85_000, session_start)
        assert d.action == "skip_pending"

    def test_fresh_session_grace_then_fire(self) -> None:
        """New session: grace period blocks, then fires after grace expires."""
        trigger = CompactionTrigger(
            threshold=70_000,
            cooldown_seconds=120,
            grace_period_seconds=10,
        )

        # Session just started
        session_start = time.time()
        d = trigger.evaluate(80_000, session_start)
        assert d.action == "skip_grace_period"

        # Simulate time passing past grace period
        old_start = time.time() - 15  # 15s ago, grace is 10s
        d = trigger.evaluate(80_000, old_start)
        assert d.action == "fire"

    def test_cooldown_expiry_allows_refire(self) -> None:
        """After cooldown expires and compaction not detected, re-fires."""
        trigger = CompactionTrigger(
            threshold=70_000,
            cooldown_seconds=120,
            grace_period_seconds=0,
        )
        session_start = time.time() - 1000

        # Fire and record
        d = trigger.evaluate(80_000, session_start)
        assert d.action == "fire"
        trigger.record_reminder_sent()

        # Pending
        d = trigger.evaluate(80_000, session_start)
        assert d.action == "skip_pending"

        # Simulate cooldown expiry (pending + expired cooldown -> clears pending and fires)
        trigger._last_reminder_time = time.time() - 200

        d = trigger.evaluate(80_000, session_start)
        assert d.action == "fire"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_threshold_of_one(self) -> None:
        """Threshold of 1 token fires on 2 tokens."""
        trigger = CompactionTrigger(
            threshold=1, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        d = trigger.evaluate(1, session_start)
        assert d.action == "skip_below_threshold"

        d = trigger.evaluate(2, session_start)
        assert d.action == "fire"

    def test_reason_is_populated(self) -> None:
        """All decisions have non-empty reason strings."""
        trigger = CompactionTrigger(
            threshold=70_000, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        # Below threshold
        d = trigger.evaluate(50_000, session_start)
        assert len(d.reason) > 0

        # Fire
        d = trigger.evaluate(80_000, session_start)
        assert len(d.reason) > 0
        trigger.record_reminder_sent()

        # Pending
        d = trigger.evaluate(80_000, session_start)
        assert len(d.reason) > 0

    def test_multiple_compaction_cycles(self) -> None:
        """Multiple fire-compact cycles work correctly."""
        trigger = CompactionTrigger(
            threshold=70_000, cooldown_seconds=0, grace_period_seconds=0
        )
        session_start = time.time() - 1000

        for _ in range(5):
            d = trigger.evaluate(80_000, session_start)
            assert d.action == "fire"
            trigger.record_reminder_sent()
            trigger.record_compaction_detected()

    def test_decision_contains_correct_tokens_and_threshold(self) -> None:
        """Every decision carries the evaluated token count and threshold."""
        trigger = CompactionTrigger(threshold=70_000, grace_period_seconds=0)
        session_start = time.time() - 1000

        for tokens in [0, 50_000, 70_000, 70_001, 100_000]:
            d = trigger.evaluate(tokens, session_start)
            assert d.current_tokens == tokens
            assert d.threshold == 70_000
