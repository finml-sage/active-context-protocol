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
    """Tests for FileTracker.check_rotation."""

    def test_no_previous_session(self, tmp_path: Path) -> None:
        """First call with no prior session -> False."""
        _make_session_file(tmp_path)

        tracker = FileTracker(claude_dir=tmp_path)
        assert tracker.check_rotation() is False

    def test_same_session_no_rotation(self, tmp_path: Path) -> None:
        """Same file active on consecutive calls -> False."""
        _make_session_file(tmp_path)

        tracker = FileTracker(claude_dir=tmp_path)
        tracker.find_active_session()  # establish baseline
        assert tracker.check_rotation() is False

    def test_rotation_detected(self, tmp_path: Path) -> None:
        """New file appears with newer mtime -> rotation detected."""
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=-10)

        tracker = FileTracker(claude_dir=tmp_path)
        tracker.find_active_session()

        # Simulate new session appearing
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100)

        assert tracker.check_rotation() is True

    def test_rotation_resets_read_position(self, tmp_path: Path) -> None:
        """Rotation resets the read position to 0."""
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=-10)

        tracker = FileTracker(claude_dir=tmp_path)
        tracker.find_active_session()
        tracker.update_read_position(500)
        assert tracker.get_read_position() == 500

        # New session
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100)
        tracker.check_rotation()

        assert tracker.get_read_position() == 0

    def test_no_files_no_rotation(self, tmp_path: Path) -> None:
        """Empty dir -> no rotation."""
        tracker = FileTracker(claude_dir=tmp_path)
        assert tracker.check_rotation() is False

    def test_rotation_after_deletion_and_new_file(self, tmp_path: Path) -> None:
        """Old file deleted, new file appears -> rotation detected."""
        old_id = str(uuid.uuid4())
        old_path = _make_session_file(tmp_path, session_id=old_id, mtime_offset=-10)

        tracker = FileTracker(claude_dir=tmp_path)
        tracker.find_active_session()

        # Delete old, create new
        old_path.unlink()
        _make_session_file(tmp_path, session_id=str(uuid.uuid4()), mtime_offset=100)

        assert tracker.check_rotation() is True


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

        # No sessions yet
        assert tracker.find_active_session() is None
        assert tracker.check_rotation() is False

        # Session 1 starts
        s1_id = str(uuid.uuid4())
        _make_session_file(tmp_path, session_id=s1_id, content='{"event":"start"}\n')

        info1 = tracker.find_active_session()
        assert info1 is not None
        assert info1.session_id == s1_id

        # Read some data
        tracker.update_read_position(19)
        assert tracker.get_read_position() == 19

        # No rotation yet
        assert tracker.check_rotation() is False

        # Session 2 starts (new file)
        s2_id = str(uuid.uuid4())
        _make_session_file(tmp_path, session_id=s2_id, content='{"event":"start2"}\n', mtime_offset=200)

        # Rotation detected
        assert tracker.check_rotation() is True
        assert tracker.get_read_position() == 0  # reset

        info2 = tracker.find_active_session()
        assert info2 is not None
        assert info2.session_id == s2_id

    def test_multiple_projects_lifecycle(self, tmp_path: Path) -> None:
        """Sessions across different projects, tracker follows the most recent."""
        tracker = FileTracker(claude_dir=tmp_path)

        # Project A session
        _make_session_file(tmp_path, project="proj-a", mtime_offset=-50)
        info = tracker.find_active_session()
        assert info is not None
        assert info.project_path == "proj-a"

        # Project B session (newer)
        _make_session_file(tmp_path, project="proj-b", mtime_offset=50)
        assert tracker.check_rotation() is True
        info = tracker.find_active_session()
        assert info is not None
        assert info.project_path == "proj-b"

        # Project A gets a newer session
        _make_session_file(tmp_path, project="proj-a", mtime_offset=200)
        assert tracker.check_rotation() is True
        info = tracker.find_active_session()
        assert info is not None
        assert info.project_path == "proj-a"
