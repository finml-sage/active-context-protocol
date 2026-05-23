"""JSONL Session File Tracker for Claude Code.

Finds and tracks the active Claude Code session JSONL file.
Claude Code stores session data at ~/.claude/projects/<project-path>/<session-id>.jsonl.
Each session gets a unique JSONL file named by UUID. When a new session starts,
a new file appears. This tracker detects the active file and rotation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CLAUDE_DIR = Path.home() / ".claude" / "projects"


@dataclass(frozen=True)
class SessionInfo:
    """Metadata about a Claude Code session JSONL file.

    Attributes:
        file_path: Absolute path to the JSONL file.
        session_id: UUID extracted from the filename (stem).
        project_path: Name of the parent directory (the project directory).
        size: Current file size in bytes.
        mtime: Last modification time (seconds since epoch).
    """

    file_path: Path
    session_id: str
    project_path: str
    size: int
    mtime: float


def _is_valid_uuid(value: str) -> bool:
    """Check whether *value* is a valid UUID string."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _stat_session(path: Path) -> SessionInfo | None:
    """Build a SessionInfo from *path*, or None if the file is gone."""
    try:
        st = path.stat()
    except OSError:
        return None
    return SessionInfo(
        file_path=path,
        session_id=path.stem,
        project_path=path.parent.name,
        size=st.st_size,
        mtime=st.st_mtime,
    )


class FileTracker:
    """Tracks the active Claude Code session JSONL file.

    The tracker scans ``claude_dir`` (defaults to ``~/.claude/projects/``)
    for ``*.jsonl`` files whose stems are valid UUIDs.  The most recently
    modified file is considered the *active* session.

    Usage::

        tracker = FileTracker()
        info = tracker.find_active_session()
        if info:
            print(f"Active session: {info.session_id} in {info.project_path}")

    The tracker also detects *rotation* — when the active session file
    changes between successive calls — and maintains a per-file read
    position so that consumers can resume incremental reads.
    """

    def __init__(self, claude_dir: Path | None = None) -> None:
        """Initialise the tracker.

        Args:
            claude_dir: Root directory to scan.  Defaults to
                ``~/.claude/projects/``.
        """
        self._claude_dir: Path = claude_dir if claude_dir is not None else _DEFAULT_CLAUDE_DIR
        self._current_session: SessionInfo | None = None
        self._read_position: int = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def claude_dir(self) -> Path:
        """The root directory being scanned."""
        return self._claude_dir

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def find_active_session(self) -> SessionInfo | None:
        """Return the most recently modified session JSONL file.

        Scans all ``*.jsonl`` files recursively under :pyattr:`claude_dir`
        whose filenames (stems) are valid UUIDs.  Returns the one with the
        highest modification time, or ``None`` if no qualifying files exist.

        This method does **not** mutate any rotation-detection state.
        Callers that need to detect rotation must pass both the previous
        and current sessions to :pymeth:`check_rotation` explicitly.
        """
        if not self._claude_dir.is_dir():
            logger.debug("claude_dir does not exist: %s", self._claude_dir)
            return None

        best: SessionInfo | None = None

        for path in self._claude_dir.rglob("*.jsonl"):
            if not _is_valid_uuid(path.stem):
                continue
            info = _stat_session(path)
            if info is None:
                continue
            if best is None or info.mtime > best.mtime:
                best = info

        return best

    def check_rotation(self, current: SessionInfo | None) -> bool:
        """Check whether the active session has rotated.

        Compares *current* (the session returned by the latest
        :pymeth:`find_active_session` call) with the previously remembered
        active session.  Returns ``True`` if the active file path has
        changed, indicating a new session started.  On the very first
        call (no previous session known) this returns ``False``.

        When rotation is detected the read position is reset to 0 and
        the remembered session is advanced to *current*.

        Args:
            current: The session most recently discovered by the caller,
                or ``None`` if no active session exists this tick.

        Note:
            Earlier versions of this method took no argument and called
            ``find_active_session()`` internally to obtain *current*.
            That was unsafe because the caller had typically already
            called ``find_active_session()`` itself; the previous tracker
            also mutated ``_current_session`` inside ``find_active_session``,
            which meant the "previous" captured here was the same object
            as "current" and rotation was never detected.  The current
            contract requires the caller to pass *current* explicitly.
        """
        previous = self._current_session

        if current is None:
            # No active session this tick — do not advance and do not
            # report rotation.  A future tick with a real session will
            # treat the prior remembered session as "previous".
            return False

        if previous is None:
            # First observation — remember it but do not report rotation.
            self._current_session = current
            return False

        rotated = current.file_path != previous.file_path
        if rotated:
            logger.info(
                "Session rotated: %s -> %s",
                previous.session_id,
                current.session_id,
            )
            self._read_position = 0
        self._current_session = current
        return rotated

    def get_read_position(self) -> int:
        """Return the last-read byte position in the current file."""
        return self._read_position

    def update_read_position(self, pos: int) -> None:
        """Set the last-read byte position in the current file.

        Args:
            pos: Byte offset (must be >= 0).

        Raises:
            ValueError: If *pos* is negative.
        """
        if pos < 0:
            raise ValueError(f"Read position must be >= 0, got {pos}")
        self._read_position = pos
