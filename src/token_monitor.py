"""Token Usage Monitor for Claude Code JSONL sessions.

Reads token usage data from Claude Code session JSONL files. Each ``assistant``
type entry contains a ``message.usage`` block with per-call token counts.
These counts are NOT cumulative -- each entry reports the full context window
usage for that API call. The monitor finds the latest assistant entry's usage
and computes total context consumption.

Formula::

    total_context = input_tokens + cache_creation_input_tokens + cache_read_input_tokens

This value grows monotonically over the session, so reading the latest entry
is sufficient to determine current context usage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Immutable snapshot of token usage from a single assistant entry.

    Attributes:
        input_tokens: Direct input tokens for the API call.
        cache_creation_input_tokens: Tokens used to create the prompt cache.
        cache_read_input_tokens: Tokens read from the prompt cache.
        output_tokens: Tokens generated as output.
        total_context: Computed context window usage
            (input + cache_creation + cache_read).
        timestamp: ISO timestamp from the JSONL entry, or empty string if
            the entry had no timestamp.
    """

    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    total_context: int
    timestamp: str


def _parse_usage(entry: dict) -> TokenUsage | None:
    """Extract a TokenUsage from a parsed JSONL entry, or None if missing."""
    if entry.get("type") != "assistant":
        return None

    message = entry.get("message")
    if not isinstance(message, dict):
        return None

    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None

    try:
        input_tokens = int(usage.get("input_tokens", 0))
        cache_creation = int(usage.get("cache_creation_input_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
    except (TypeError, ValueError):
        return None

    total_context = input_tokens + cache_creation + cache_read
    timestamp = entry.get("timestamp", "")
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)

    return TokenUsage(
        input_tokens=input_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        output_tokens=output_tokens,
        total_context=total_context,
        timestamp=timestamp,
    )


class TokenMonitor:
    """Monitors token usage from Claude Code session JSONL files.

    Reads JSONL files incrementally (from a given byte position), finds the
    latest ``assistant`` entry with ``message.usage`` data, and reports
    whether the context window has exceeded a configurable threshold.

    Usage::

        monitor = TokenMonitor(threshold=70_000)
        usage = monitor.read_latest_usage(Path("session.jsonl"))
        if usage and monitor.is_above_threshold(usage):
            print(f"Context at {usage.total_context} tokens — above threshold!")

    Args:
        threshold: Context token count above which
            :meth:`is_above_threshold` returns True.  Default 70,000.
    """

    def __init__(self, threshold: int = 70_000) -> None:
        self._threshold = threshold
        self._last_position: int = 0

    @property
    def threshold(self) -> int:
        """The context token threshold."""
        return self._threshold

    def read_latest_usage(
        self, jsonl_path: Path, from_position: int = 0
    ) -> TokenUsage | None:
        """Read the latest token usage from a JSONL session file.

        Opens *jsonl_path*, seeks to *from_position*, and reads all lines
        from that point forward. Parses each line as JSON and looks for
        ``assistant`` type entries containing ``message.usage``. Returns the
        usage from the **last** such entry (the most recent API call), or
        ``None`` if no usage data is found.

        Malformed lines are silently skipped with a debug log message.

        Args:
            jsonl_path: Path to the Claude Code session JSONL file.
            from_position: Byte offset to start reading from.  Use 0 to
                read the entire file, or pass the value from
                :meth:`get_new_position` for incremental reads.

        Returns:
            A :class:`TokenUsage` with the latest usage data, or ``None``
            if no assistant entries with usage were found, the file is
            empty, or the file does not exist.
        """
        if not jsonl_path.is_file():
            logger.debug("JSONL file does not exist: %s", jsonl_path)
            self._last_position = from_position
            return None

        latest: TokenUsage | None = None

        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                f.seek(from_position)
                while True:
                    line = f.readline()
                    if not line:
                        # EOF
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

                    usage = _parse_usage(entry)
                    if usage is not None:
                        latest = usage

        except OSError as exc:
            logger.warning("Error reading JSONL file %s: %s", jsonl_path, exc)
            self._last_position = from_position
            return None

        return latest

    def is_above_threshold(self, usage: TokenUsage) -> bool:
        """Check whether the usage exceeds the configured threshold.

        Args:
            usage: A :class:`TokenUsage` instance.

        Returns:
            True if ``usage.total_context`` exceeds the threshold.
        """
        return usage.total_context > self._threshold

    def get_new_position(self) -> int:
        """Return the byte position after the last :meth:`read_latest_usage` call.

        Use this value as the ``from_position`` argument on the next call
        to :meth:`read_latest_usage` for incremental reads.

        Returns:
            Byte offset in the file after the last read completed.
        """
        return self._last_position
