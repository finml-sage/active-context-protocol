"""Comprehensive tests for the JSONL Session File Tracker."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from src.file_tracker import FileTracker, SessionInfo, _is_valid_uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_file(
    base: Path,
    project: str = "-home-user-myproject",
    session_id: str | None = None,
    content: str = "",
    mtime_offset: float = 0.0,
) -> Path:
    """Create a fake session JSONL file and return its path.

    Args:
        base: The claude_dir root.
        project: Project directory name.
        session_id: UUID string for filename; auto-generated if None.
        content: File content to write.
        mtime_offset: Seconds to add to the current time for mtime.
    """
    sid = session_id or str(uuid.uuid4())
    project_dir = base / project
    project_dir.mkdir(parents=True, exist_ok=True)
    fpath = project_dir / f"{sid}.jsonl"
    fpath.write_text(content)
    if mtime_offset != 0.0:
        st = fpath.stat()
        os.utime(fpath, (st.st_atime, st.st_mtime + mtime_offset))
    return fpath


# ---------------------------------------------------------------------------
# SessionInfo dataclass
# ---------------------------------------------------------------------------

class TestSessionInfo:
    """Tests for the SessionInfo dataclass."""

    def test_fields(self, tmp_path: Path) -> None:
        info = SessionInfo(
            file_path=tmp_path / "test.jsonl",
            session_id="abc-123",
            project_path="myproject",
            size=42,
            mtime=1000.0,
        )
        assert info.file_path == tmp_path / "test.jsonl"
        assert info.session_id == "abc-123"
        assert info.project_path == "myproject"
        assert info.size == 42
        assert info.mtime == 1000.0

    def test_frozen(self, tmp_path: Path) -> None:
        info = SessionInfo(
            file_path=tmp_path / "test.jsonl",
            session_id="abc",
            project_path="proj",
            size=0,
            mtime=0.0,
        )
        with pytest.raises(AttributeError):
            info.size = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _is_valid_uuid helper
# ---------------------------------------------------------------------------

class TestIsValidUuid:
    def test_valid_uuid4(self) -> None:
        assert _is_valid_uuid(str(uuid.uuid4())) is True

    def test_valid_uuid_formatted(self) -> None:
        assert _is_valid_uuid("550e8400-e29b-41d4-a716-446655440000") is True

    def test_invalid(self) -> None:
        assert _is_valid_uuid("not-a-uuid") is False

    def test_empty(self) -> None:
        assert _is_valid_uuid("") is False


# ---------------------------------------------------------------------------
# FileTracker — find_active_session
# ---------------------------------------------------------------------------

class TestFindActiveSession:
    """Tests for FileTracker.find_active_session."""

    def test_no_directory(self, tmp_path: Path) -> None:
        """claude_dir doesn't exist -> None."""
        tracker = FileTracker(claude_dir=tmp_path / "nonexistent")
        assert tracker.find_active_session() is None

    def test_empty_directory(self, tmp_path: Path) -> None:
        """claude_dir exists but has no JSONL files -> None."""
        tracker = FileTracker(claude_dir=tmp_path)
        assert tracker.find_active_session() is None

    def test_single_file(self, tmp_path: Path) -> None:
        """Single session file is returned."""
        sid = str(uuid.uuid4())
        fpath = _make_session_file(tmp_path, session_id=sid, content='{"type":"init"}\n')
        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()

        assert info is not None
        assert info.file_path == fpath
        assert info.session_id == sid
        assert info.project_path == "-home-user-myproject"
        assert info.size == len('{"type":"init"}\n')
        assert info.mtime > 0

    def test_most_recent_wins(self, tmp_path: Path) -> None:
        """Among multiple files, the most recently modified wins."""
        _make_session_file(tmp_path, project="proj-a", mtime_offset=-100)
        newest = _make_session_file(tmp_path, project="proj-b", mtime_offset=100)

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()

        assert info is not None
        assert info.file_path == newest

    def test_multiple_projects(self, tmp_path: Path) -> None:
        """Files across multiple project dirs are all considered."""
        _make_session_file(tmp_path, project="proj-1", mtime_offset=-50)
        _make_session_file(tmp_path, project="proj-2", mtime_offset=-25)
        newest = _make_session_file(tmp_path, project="proj-3", mtime_offset=50)

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()

        assert info is not None
        assert info.file_path == newest
        assert info.project_path == "proj-3"

    def test_ignores_non_uuid_filenames(self, tmp_path: Path) -> None:
        """Files whose stems are not valid UUIDs are ignored."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "not-a-uuid.jsonl").write_text("{}\n")
        (project_dir / "random.jsonl").write_text("{}\n")

        tracker = FileTracker(claude_dir=tmp_path)
        assert tracker.find_active_session() is None

    def test_uuid_file_among_non_uuid(self, tmp_path: Path) -> None:
        """A valid UUID file is found even when non-UUID files exist."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "config.jsonl").write_text("{}\n")
        sid = str(uuid.uuid4())
        expected = project_dir / f"{sid}.jsonl"
        expected.write_text('{"hello":"world"}\n')

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()

        assert info is not None
        assert info.session_id == sid

    def test_deeply_nested(self, tmp_path: Path) -> None:
        """Files in nested sub-directories are found (rglob)."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        sid = str(uuid.uuid4())
        fpath = deep / f"{sid}.jsonl"
        fpath.write_text("{}\n")

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()

        assert info is not None
        assert info.file_path == fpath

    def test_agent_offline_returns_last_known(self, tmp_path: Path) -> None:
        """When the agent stops writing, the file is still returned."""
        _make_session_file(tmp_path, mtime_offset=-3600)  # 1 hour old

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()

        assert info is not None  # still found, even though old

    def test_default_claude_dir(self) -> None:
        """Default claude_dir points to ~/.claude/projects/."""
        tracker = FileTracker()
        assert tracker.claude_dir == Path.home() / ".claude" / "projects"


# ---------------------------------------------------------------------------
# FileTracker — check_rotation
# ---------------------------------------------------------------------------

class TestCheckRotation:
    """Tests for FileTracker.check_rotation.

    Note: ``check_rotation`` requires the caller to pass in the session
    most recently returned by ``find_active_session``.  Each test below
    mirrors the daemon's tick pattern: find-then-check.
    """

    def test_no_previous_session(self, tmp_path: Path) -> None:
        """First call with no prior session -> False."""
        _make_session_file(tmp_path)

        tracker = FileTracker(claude_dir=tmp_path)
        current = tracker.find_active_session()
        assert tracker.check_rotation(current) is False

    def test_same_session_no_rotation(self, tmp_path: Path) -> None:
        """Same file active on consecutive calls -> False."""
        _make_session_file(tmp_path)

        tracker = FileTracker(claude_dir=tmp_path)
        # Tick 1 — establish baseline.
        tracker.check_rotation(tracker.find_active_session())
        # Tick 2 — same file.
        assert tracker.check_rotation(tracker.find_active_session()) is False

    def test_rotation_detected(self, tmp_path: Path) -> None:
        """New file appears with newer mtime -> rotation detected."""
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=-10)

        tracker = FileTracker(claude_dir=tmp_path)
        # Tick 1 — establish baseline.
        tracker.check_rotation(tracker.find_active_session())

        # Simulate new session appearing.
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100)

        # Tick 2 — rotation expected.
        assert tracker.check_rotation(tracker.find_active_session()) is True

    def test_rotation_resets_read_position(self, tmp_path: Path) -> None:
        """Rotation resets the read position to 0."""
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=-10)

        tracker = FileTracker(claude_dir=tmp_path)
        tracker.check_rotation(tracker.find_active_session())
        tracker.update_read_position(500)
        assert tracker.get_read_position() == 500

        # New session.
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100)
        tracker.check_rotation(tracker.find_active_session())

        assert tracker.get_read_position() == 0

    def test_no_files_no_rotation(self, tmp_path: Path) -> None:
        """Empty dir -> no rotation."""
        tracker = FileTracker(claude_dir=tmp_path)
        assert tracker.check_rotation(tracker.find_active_session()) is False

    def test_rotation_after_deletion_and_new_file(self, tmp_path: Path) -> None:
        """Old file deleted, new file appears -> rotation detected."""
        old_id = str(uuid.uuid4())
        old_path = _make_session_file(tmp_path, session_id=old_id, mtime_offset=-10)

        tracker = FileTracker(claude_dir=tmp_path)
        tracker.check_rotation(tracker.find_active_session())

        # Delete old, create new.
        old_path.unlink()
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100)

        assert tracker.check_rotation(tracker.find_active_session()) is True

    def test_check_rotation_requires_argument(self, tmp_path: Path) -> None:
        """Calling check_rotation() with no argument is a TypeError.

        Guards against regressions to the prior (broken) zero-arg form
        which silently swallowed rotations.
        """
        tracker = FileTracker(claude_dir=tmp_path)
        with pytest.raises(TypeError):
            tracker.check_rotation()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# FileTracker — read position
# ---------------------------------------------------------------------------

class TestReadPosition:
    """Tests for read position tracking."""

    def test_initial_position_is_zero(self, tmp_path: Path) -> None:
        tracker = FileTracker(claude_dir=tmp_path)
        assert tracker.get_read_position() == 0

    def test_update_and_get(self, tmp_path: Path) -> None:
        tracker = FileTracker(claude_dir=tmp_path)
        tracker.update_read_position(1024)
        assert tracker.get_read_position() == 1024

    def test_update_multiple_times(self, tmp_path: Path) -> None:
        tracker = FileTracker(claude_dir=tmp_path)
        tracker.update_read_position(100)
        tracker.update_read_position(200)
        tracker.update_read_position(300)
        assert tracker.get_read_position() == 300

    def test_negative_position_raises(self, tmp_path: Path) -> None:
        tracker = FileTracker(claude_dir=tmp_path)
        with pytest.raises(ValueError, match="must be >= 0"):
            tracker.update_read_position(-1)

    def test_zero_position_allowed(self, tmp_path: Path) -> None:
        tracker = FileTracker(claude_dir=tmp_path)
        tracker.update_read_position(100)
        tracker.update_read_position(0)
        assert tracker.get_read_position() == 0


# ---------------------------------------------------------------------------
# FileTracker — file deletion during tracking
# ---------------------------------------------------------------------------

class TestFileDeletionDuringTracking:
    """Tests for graceful handling when files disappear."""

    def test_active_file_deleted(self, tmp_path: Path) -> None:
        """Active file deleted -> find_active_session returns None (no other files)."""
        sid = str(uuid.uuid4())
        fpath = _make_session_file(tmp_path, session_id=sid)

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()
        assert info is not None

        # Delete it
        fpath.unlink()

        info2 = tracker.find_active_session()
        # No files left, but _current_session still holds the last known
        assert info2 is None

    def test_active_file_deleted_fallback_to_another(self, tmp_path: Path) -> None:
        """Active file deleted but another exists -> returns the other."""
        older = _make_session_file(tmp_path, project="p1", mtime_offset=-100)
        newer = _make_session_file(tmp_path, project="p2", mtime_offset=100)

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()
        assert info is not None
        assert info.file_path == newer

        newer.unlink()

        info2 = tracker.find_active_session()
        assert info2 is not None
        assert info2.file_path == older

    def test_directory_deleted(self, tmp_path: Path) -> None:
        """Entire claude_dir deleted -> returns None."""
        import shutil

        _make_session_file(tmp_path, project="proj")

        tracker = FileTracker(claude_dir=tmp_path)
        info = tracker.find_active_session()
        assert info is not None

        shutil.rmtree(tmp_path)

        assert tracker.find_active_session() is None


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple tracker features."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Simulate a full session lifecycle: start, read, rotate, read."""
        tracker = FileTracker(claude_dir=tmp_path)

        # No sessions yet.
        assert tracker.find_active_session() is None
        assert tracker.check_rotation(None) is False

        # Session 1 starts.
        s1_id = str(uuid.uuid4())
        _make_session_file(tmp_path, session_id=s1_id, content='{"event":"start"}\n')

        info1 = tracker.find_active_session()
        assert info1 is not None
        assert info1.session_id == s1_id

        # First observation — no rotation, baseline remembered.
        assert tracker.check_rotation(info1) is False

        # Read some data.
        tracker.update_read_position(19)
        assert tracker.get_read_position() == 19

        # Same session on next tick — no rotation.
        assert tracker.check_rotation(tracker.find_active_session()) is False

        # Session 2 starts (new file).
        s2_id = str(uuid.uuid4())
        _make_session_file(tmp_path, session_id=s2_id, content='{"event":"start2"}\n', mtime_offset=200)

        info2 = tracker.find_active_session()
        assert info2 is not None
        assert info2.session_id == s2_id

        # Rotation detected; read position resets.
        assert tracker.check_rotation(info2) is True
        assert tracker.get_read_position() == 0

    def test_multiple_projects_lifecycle(self, tmp_path: Path) -> None:
        """Sessions across different projects, tracker follows the most recent."""
        tracker = FileTracker(claude_dir=tmp_path)

        # Project A session.
        _make_session_file(tmp_path, project="proj-a", mtime_offset=-50)
        info = tracker.find_active_session()
        assert info is not None
        assert info.project_path == "proj-a"
        assert tracker.check_rotation(info) is False  # baseline

        # Project B session (newer).
        _make_session_file(tmp_path, project="proj-b", mtime_offset=50)
        info = tracker.find_active_session()
        assert info is not None
        assert info.project_path == "proj-b"
        assert tracker.check_rotation(info) is True

        # Project A gets a newer session.
        _make_session_file(tmp_path, project="proj-a", mtime_offset=200)
        info = tracker.find_active_session()
        assert info is not None
        assert info.project_path == "proj-a"
        assert tracker.check_rotation(info) is True


# ---------------------------------------------------------------------------
# Regression: daemon main-loop call pattern (find then check)
# ---------------------------------------------------------------------------

class TestDaemonCallPatternRotation:
    """Regression tests for the daemon's actual call pattern.

    The daemon's main loop (src/cli.py) calls ``find_active_session()`` on
    every iteration BEFORE calling ``check_rotation()``.  A prior version of
    ``check_rotation()`` captured ``self._current_session`` as the "previous"
    session, but ``find_active_session()`` had already overwritten that field
    moments earlier — so ``previous`` and ``current`` were always identical
    and rotation was never detected.

    Empirical evidence of the silent failure: the delivery audit log showed
    no fires between 2026-05-13 and 2026-05-23 even though multiple
    Claude Code sessions on May 21 and May 23 crossed the 250k-token
    threshold.  The root cause was the rotation check above.
    """

    def test_rotation_detected_under_daemon_call_pattern(self, tmp_path: Path) -> None:
        """find_active_session() then check_rotation(session) must detect rotation.

        Replicates ``src/cli.py::run_monitor``: every polling tick calls
        ``find_active_session()`` first and then ``check_rotation(session)``.
        """
        tracker = FileTracker(claude_dir=tmp_path)

        # Tick 1: only session 1 exists.
        _make_session_file(
            tmp_path, session_id=str(uuid.uuid4()), mtime_offset=-10
        )
        session_1 = tracker.find_active_session()
        assert session_1 is not None
        assert tracker.check_rotation(session_1) is False  # no prior session

        # Tick 2: session 2 appears with a newer mtime — rotation expected.
        _make_session_file(
            tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100
        )
        session_2 = tracker.find_active_session()
        assert session_2 is not None
        assert session_2.file_path != session_1.file_path

        # Pre-fix, check_rotation took no argument and re-called
        # find_active_session() internally, which (because the prior
        # find_active_session on the line above had already mutated
        # _current_session) caused "previous" == "current" and rotation
        # was silently dropped.  The fix is to pass *current* in
        # explicitly so the comparison uses the truly-previous session.
        assert tracker.check_rotation(session_2) is True

    def test_rotation_resets_read_position_under_daemon_call_pattern(
        self, tmp_path: Path
    ) -> None:
        """Read position must reset to 0 when rotation fires under the
        daemon's call order.

        Consequence of the bug pre-fix: ``TokenMonitor._last_position`` /
        ``FileTracker._read_position`` leaked across sessions, so the next
        ``f.seek(leaked_offset)`` landed past EOF on the (smaller) new
        JSONL and ``read_latest_usage`` returned ``None`` — no reminder
        could fire even when the new session crossed the threshold.
        """
        tracker = FileTracker(claude_dir=tmp_path)

        # Tick 1: session 1 exists; daemon advances read position into it.
        _make_session_file(
            tmp_path, session_id=str(uuid.uuid4()), mtime_offset=-10
        )
        s1 = tracker.find_active_session()
        tracker.check_rotation(s1)  # no prior session -> False
        tracker.update_read_position(500_000)
        assert tracker.get_read_position() == 500_000

        # Tick 2: session 2 appears (smaller, newer). Daemon polls.
        _make_session_file(
            tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100
        )
        s2 = tracker.find_active_session()
        rotated = tracker.check_rotation(s2)

        assert rotated is True
        assert tracker.get_read_position() == 0

    def test_no_rotation_under_daemon_call_pattern(self, tmp_path: Path) -> None:
        """When the active file is unchanged across two ticks, the daemon
        call pattern must report no rotation."""
        tracker = FileTracker(claude_dir=tmp_path)

        _make_session_file(tmp_path, session_id=str(uuid.uuid4()))

        # Tick 1.
        tick_1 = tracker.find_active_session()
        tracker.check_rotation(tick_1)

        # Tick 2 — no new file.
        tick_2 = tracker.find_active_session()
        assert tracker.check_rotation(tick_2) is False
