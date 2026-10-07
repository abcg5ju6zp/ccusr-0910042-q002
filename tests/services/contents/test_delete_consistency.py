"""Tests for recoverable consistency of content deletion.

A delete spans two stores -- the content (file/directory, possibly sent to the
trash) and its checkpoints -- and ends with an event. These tests assert that a
failure at any stage never leaves the three disagreeing:

* content gone while an old checkpoint is still visible (or vice versa),
* a delete event describing something that did not actually finish,
* a retried request that cannot tell whether to keep cleaning up or restore.

Both the synchronous and asynchronous file managers are exercised through the
same fixtures as ``test_manager.py``.
"""

import ast
import inspect
import logging
import os

import pytest
from jupyter_core.utils import ensure_async
from tornado.web import HTTPError

from jupyter_server.services.contents.filecheckpoints import FileCheckpoints
from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)

from .test_manager import new_notebook


class _DeleteEventCollector(logging.Handler):
    """Collect ``action`` values emitted by the contents event logger."""

    def __init__(self):
        super().__init__()
        self.actions = []

    def emit(self, record):
        try:
            data = ast.literal_eval(record.getMessage())
        except (ValueError, SyntaxError):
            return
        if isinstance(data, dict) and "action" in data:
            self.actions.append(data["action"])

    @property
    def deletes(self):
        return [action for action in self.actions if action == "delete"]


def _raise_permission_error(*args, **kwargs):
    raise PermissionError("boom")


@pytest.fixture(
    params=[
        (FileContentsManager, True),
        (FileContentsManager, False),
        (AsyncFileContentsManager, True),
        (AsyncFileContentsManager, False),
    ]
)
def jp_contents_manager(request, tmp_path):
    """Same parametrisation as test_manager.py: sync/async x atomic writes."""
    contents_manager, use_atomic_writing = request.param
    return contents_manager(root_dir=str(tmp_path), use_atomic_writing=use_atomic_writing)


@pytest.fixture
def cm(jp_contents_manager):
    """A contents manager with delete events captured."""
    manager = jp_contents_manager
    collector = _DeleteEventCollector()
    manager.event_logger.register_handler(collector)
    manager.delete_to_trash = False
    manager._event_collector = collector
    yield manager
    manager.event_logger.remove_handler(collector)


async def _notebook_with_checkpoint(manager):
    _nb, _name, path = await new_notebook(manager)
    # Saving a new notebook already creates one checkpoint, but make the
    # expectation explicit and independent of that default behaviour.
    await ensure_async(manager.create_checkpoint(path))
    assert await ensure_async(manager.list_checkpoints(path))
    return path


async def _make_dir(manager, api_path):
    await ensure_async(manager.save({"type": "directory"}, api_path))


def _content_paths(manager, path):
    os_path = manager._get_os_path(path)
    cp_path = manager.checkpoints._primary_checkpoint_path(path)
    return os_path, cp_path


def _assert_no_transaction_leftovers(manager):
    assert not [name for name in os.listdir(manager.root_dir) if name.startswith(".jupyter")]


# ---------------------------------------------------------------------------
# Happy path and duplicate requests
# ---------------------------------------------------------------------------


async def test_delete_removes_content_and_checkpoints(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)

    await ensure_async(cm.delete(path))

    assert not os.path.exists(os_path)
    assert not os.path.exists(cp_path)
    assert not await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == ["delete"]
    _assert_no_transaction_leftovers(cm)


async def test_duplicate_delete_is_404_and_emits_once(cm):
    path = await _notebook_with_checkpoint(cm)

    await ensure_async(cm.delete(path))
    with pytest.raises(HTTPError) as exc_info:
        await ensure_async(cm.delete(path))
    assert exc_info.value.status_code == 404

    assert cm._event_collector.deletes == ["delete"]


async def test_duplicate_delete_during_retry_window_converges(cm):
    """A retry while staged content awaits cleanup finishes the cleanup."""
    path = await _notebook_with_checkpoint(cm)

    # Stage everything but stop just before committing, as if the first
    # request died after staging.
    token = cm._begin_delete_transaction(path)
    os_path, cp_path = _content_paths(cm, path)
    assert not os.path.exists(os_path)
    assert not await ensure_async(cm.list_checkpoints(path))

    # The retry resumes and commits the staged transaction instead of
    # reporting the file as alive or resurrecting it.
    await ensure_async(cm.delete(path))

    assert not os.path.exists(os_path)
    assert not os.path.exists(cp_path)
    assert not os.path.exists(token["staging"])
    assert cm._event_collector.deletes == ["delete"]


# ---------------------------------------------------------------------------
# Checkpoint cleanup failure (the reported incident): content comes back
# ---------------------------------------------------------------------------


async def test_checkpoint_cleanup_failure_restores_everything(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)

    with pytest.raises(HTTPError), pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FileCheckpoints, "discard_staged_checkpoints", _raise_permission_error)
        await ensure_async(cm.delete(path))

    # The interface returned failure; the visible fact must be the original.
    assert os.path.isfile(os_path)
    assert os.path.isfile(cp_path)
    assert await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == []

    # Retry can unambiguously finish the operation.
    await ensure_async(cm.delete(path))
    assert not os.path.exists(os_path)
    assert not await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == ["delete"]


async def test_checkpoint_cleanup_failure_restores_directory(cm):
    cm.always_delete_dir = True
    await _make_dir(cm, "to_delete")
    await ensure_async(
        cm.save({"type": "file", "format": "text", "content": "hi"}, "to_delete/f.txt")
    )
    await ensure_async(cm.create_checkpoint("to_delete/f.txt"))
    dir_path = cm._get_os_path("to_delete")
    assert os.path.isdir(dir_path)

    with pytest.raises(HTTPError), pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FileCheckpoints, "discard_staged_checkpoints", _raise_permission_error)
        await ensure_async(cm.delete("to_delete"))

    # The whole directory tree, including the nested checkpoint, is back.
    assert os.path.isfile(os.path.join(dir_path, "f.txt"))
    assert await ensure_async(cm.list_checkpoints("to_delete/f.txt"))
    assert cm._event_collector.deletes == []

    await ensure_async(cm.delete("to_delete"))
    assert not os.path.exists(dir_path)
    assert cm._event_collector.deletes == ["delete"]


async def test_checkpoint_staging_failure_never_touches_content(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)

    with pytest.raises(HTTPError), pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FileCheckpoints, "stage_purge_checkpoints", _raise_permission_error)
        await ensure_async(cm.delete(path))

    assert os.path.isfile(os_path)
    assert os.path.isfile(cp_path)
    assert await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == []


# ---------------------------------------------------------------------------
# Trash failures must behave like a failed permanent delete
# ---------------------------------------------------------------------------


async def test_trash_failure_restores_content_and_checkpoints(cm):
    cm.delete_to_trash = True
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)

    def _fail_send2trash(target):
        raise OSError("trash unavailable")

    with pytest.raises(HTTPError), pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "jupyter_server.services.contents.filemanager.send2trash",
            _fail_send2trash,
        )
        await ensure_async(cm.delete(path))

    assert os.path.isfile(os_path)
    assert os.path.isfile(cp_path)
    assert await ensure_async(cm.list_checkpoints(path))
    _assert_no_transaction_leftovers(cm)
    assert cm._event_collector.deletes == []

    # The user is free to fall back to a permanent delete on retry.
    cm.delete_to_trash = False
    await ensure_async(cm.delete(path))
    assert not os.path.exists(os_path)
    assert cm._event_collector.deletes == ["delete"]


# ---------------------------------------------------------------------------
# Compensation failure: converge to a finished delete on the next request
# ---------------------------------------------------------------------------


async def test_failed_compensation_converges_on_retry(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)
    original_restore = FileCheckpoints.restore_staged_checkpoints
    calls = {"count": 0}

    def flaky_restore(self, moves):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("cannot restore checkpoints")
        return original_restore(self, moves)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FileCheckpoints, "discard_staged_checkpoints", _raise_permission_error)
        monkeypatch.setattr(FileCheckpoints, "restore_staged_checkpoints", flaky_restore)
        with pytest.raises(HTTPError):
            await ensure_async(cm.delete(path))

    # Content and checkpoints disagree with the original view, but the
    # transaction is persisted: the content is hidden rather than half-alive.
    assert not os.path.exists(os_path)
    assert cm._event_collector.deletes == []

    # Retry needs no knowledge of what failed; it resumes and converges.
    await ensure_async(cm.delete(path))
    assert not os.path.exists(os_path)
    assert not os.path.exists(cp_path)
    assert not await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == ["delete"]


# ---------------------------------------------------------------------------
# The original bug: content vanished, checkpoint cleanup failed
# ---------------------------------------------------------------------------


async def test_retry_resumes_when_content_already_gone(cm):
    """Sequential ordering: content gone, checkpoint purge failed."""
    path = await _notebook_with_checkpoint(cm)
    os_path, _cp_path = _content_paths(cm, path)

    # Force the generic (non-transactional) ordering used by other backends and
    # make the first checkpoint purge fail on the concrete checkpoint class --
    # the async checkpoint class overrides purge_checkpoints, so patching the
    # base class would not affect it. This reproduces the reported situation.
    cm._delete_transactions_supported = False
    checkpoint_cls = type(cm.checkpoints)
    is_async = inspect.iscoroutinefunction(checkpoint_cls.purge_checkpoints)

    def fail_purge_sync(self, p):
        raise HTTPError(500, "checkpoint store down")

    async def fail_purge_async(self, p):
        raise HTTPError(500, "checkpoint store down")

    with pytest.raises(HTTPError), pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            checkpoint_cls,
            "purge_checkpoints",
            fail_purge_async if is_async else fail_purge_sync,
        )
        await ensure_async(cm.delete(path))

    # Content is gone but the old checkpoint is still visible...
    assert not os.path.exists(os_path)
    assert await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == []

    # ...and the retry recognises the state and finishes the cleanup instead
    # of either 404-ing or pretending the content is alive.
    await ensure_async(cm.delete(path))
    assert not await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == ["delete"]


async def test_content_deleted_out_of_band_resumes_checkpoint_cleanup(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, _cp_path = _content_paths(cm, path)

    os.unlink(os_path)  # content vanished, checkpoint record remained

    await ensure_async(cm.delete(path))

    assert not await ensure_async(cm.list_checkpoints(path))
    assert cm._event_collector.deletes == ["delete"]


# ---------------------------------------------------------------------------
# Crash recovery with persisted transaction state
# ---------------------------------------------------------------------------


async def test_recovery_from_state_file_after_crash(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)

    token = cm._begin_delete_transaction(path)
    assert os.path.isfile(token["state_path"])

    # Simulate a fresh request after the process died between staging and
    # commit: the state file is all that is needed to finish.
    await ensure_async(cm.delete(path))

    assert not os.path.exists(os_path)
    assert not os.path.exists(cp_path)
    assert not os.path.exists(token["state_path"])
    assert not os.path.exists(token["staging"])
    assert cm._event_collector.deletes == ["delete"]


async def test_recovery_from_orphan_staging_without_state_file(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, cp_path = _content_paths(cm, path)

    token = cm._begin_delete_transaction(path)
    os.unlink(token["state_path"])  # state file never reached disk / removed

    await ensure_async(cm.delete(path))

    assert not os.path.exists(os_path)
    assert not os.path.exists(cp_path)
    assert not os.path.exists(token["staging"])
    assert cm._event_collector.deletes == ["delete"]


async def test_recovery_does_not_clobber_recreated_content(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, _cp_path = _content_paths(cm, path)

    token = cm._begin_delete_transaction(path)

    # While the old transaction is staged, the user recreates the path.
    await ensure_async(cm.save({"type": "file", "format": "text", "content": "new"}, path))
    assert os.path.isfile(os_path)

    # Recovery must abandon the old transaction instead of deleting the new
    # content or its checkpoints.
    recovered = cm._recover_pending_delete(path)
    assert recovered is False
    assert os.path.isfile(os_path)
    assert not os.path.exists(token["staging"])

    # A normal delete of the recreated path now succeeds.
    await ensure_async(cm.delete(path))
    assert not os.path.exists(os_path)
    assert cm._event_collector.deletes == ["delete"]


# ---------------------------------------------------------------------------
# Event honesty
# ---------------------------------------------------------------------------


async def test_no_delete_event_when_content_delete_fails(cm):
    path = await _notebook_with_checkpoint(cm)
    os_path, _cp_path = _content_paths(cm, path)

    # Trash mode refuses a read-only target before any removal happens.
    cm.delete_to_trash = True
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(cm, "is_writable", lambda p: False, raising=False)
        with pytest.raises(HTTPError):
            await ensure_async(cm.delete(path))

    assert os.path.isfile(os_path)
    assert cm._event_collector.deletes == []
