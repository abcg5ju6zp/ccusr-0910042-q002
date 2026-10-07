"""Tests for recoverable, consistent content + checkpoint deletion.

These tests cover the failure modes described in the delete-consistency
fix:

* files and directories, trashed and permanently deleted,
* checkpoint storage failing after the content is gone,
* content deletion failing (before and after the content is removed),
* compensation (checkpoint restore) itself failing,
* duplicate (concurrent and sequential) requests,
* the ``delete`` event being emitted only when the delete truly completed.
"""

import asyncio
import io
import json
import logging
import os
import threading

import pytest
from jupyter_core.utils import ensure_async
from nbformat.v4 import new_markdown_cell, new_notebook
from tornado.web import HTTPError

from jupyter_server.services.contents.filecheckpoints import (
    AsyncFileCheckpoints,
    FileCheckpoints,
)
from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def permanent_delete(jp_contents_manager):
    """Contents manager configured for permanent deletion."""
    cm = jp_contents_manager
    cm.delete_to_trash = False
    cm.always_delete_dir = True
    return cm


def is_async(cm):
    return isinstance(cm, AsyncFileContentsManager)


def event_sink(cm):
    """Attach a StringIO handler to the manager's event logger."""
    sink = io.StringIO()
    cm.event_logger.register_handler(logging.StreamHandler(sink))
    return sink


def recorded_events(sink):
    return [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]


def delete_events(sink):
    return [e for e in recorded_events(sink) if e.get("action") == "delete"]


async def make_file(cm, path="a.txt", content="hello\n"):
    await ensure_async(cm.save({"type": "file", "format": "text", "content": content}, path))
    await ensure_async(cm.create_checkpoint(path))
    return path


async def make_notebook(cm, path="a.ipynb"):
    nb = new_notebook(cells=[new_markdown_cell("hello")])
    await ensure_async(cm.save({"type": "notebook", "content": nb}, path))
    await ensure_async(cm.create_checkpoint(path))
    return path


async def make_dir_with_notebook(cm, dir_path="proj"):
    await ensure_async(cm.save({"type": "directory"}, dir_path))
    nb_path = f"{dir_path}/n.ipynb"
    await make_notebook(cm, nb_path)
    return dir_path, nb_path


def fail_after_unlink(cm, path, monkeypatch, message="storage error after unlink"):
    """Make delete_file remove the content and then raise."""
    os_path = cm._get_os_path(path)
    real = type(cm).delete_file

    if is_async(cm):

        async def remove_then_fail(p):
            # Call the real implementation with the instance.
            await real(cm, p)
            assert not os.path.exists(os_path)
            raise RuntimeError(message)

    else:

        def remove_then_fail(p):
            real(cm, p)
            assert not os.path.exists(os_path)
            raise RuntimeError(message)

    monkeypatch.setattr(cm, "delete_file", remove_then_fail)


def fail_before_remove(cm, monkeypatch, message="delete_file failed"):
    """Make delete_file raise without touching the content."""
    if is_async(cm):

        async def fail_delete(p):
            raise RuntimeError(message)

    else:

        def fail_delete(p):
            raise RuntimeError(message)

    monkeypatch.setattr(cm, "delete_file", fail_delete)


# ---------------------------------------------------------------------------
# happy path: files / directories, trash / permanent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_trash", [False, True])
async def test_delete_file_removes_content_and_checkpoint(jp_contents_manager, use_trash):
    cm = jp_contents_manager
    cm.delete_to_trash = use_trash
    sink = event_sink(cm)
    path = await make_notebook(cm)

    assert await ensure_async(cm.list_checkpoints(path))
    await ensure_async(cm.delete(path))

    assert not await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.list_checkpoints(path)) == []
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path)) == []
    assert len(delete_events(sink)) == 1


async def test_delete_directory_permanent_removes_nested_checkpoints(permanent_delete):
    cm = permanent_delete
    sink = event_sink(cm)
    dir_path, nb_path = await make_dir_with_notebook(cm)

    assert await ensure_async(cm.list_checkpoints(nb_path))
    await ensure_async(cm.delete(dir_path))

    assert not await ensure_async(cm.dir_exists(dir_path))
    assert await ensure_async(cm.list_checkpoints(nb_path)) == []
    assert await ensure_async(cm.checkpoints.find_delete_transactions(dir_path)) == []
    assert len(delete_events(sink)) == 1


async def test_delete_directory_to_trash_removes_nested_checkpoints(
    jp_contents_manager,
):
    cm = jp_contents_manager
    cm.delete_to_trash = True
    sink = event_sink(cm)
    dir_path, nb_path = await make_dir_with_notebook(cm)

    try:
        await ensure_async(cm.delete(dir_path))
    except HTTPError as e:
        # send2trash can legitimately refuse on cross-device setups.
        if "send2trash" in str(e).lower():
            pytest.skip("trash unavailable on this filesystem layout")
        raise

    assert not await ensure_async(cm.dir_exists(dir_path))
    assert await ensure_async(cm.list_checkpoints(nb_path)) == []
    assert len(delete_events(sink)) == 1


async def test_trash_failure_leaves_content_and_checkpoint(
    jp_contents_manager, monkeypatch
):
    """When send2trash fails the content survives and checkpoints roll back."""
    cm = jp_contents_manager
    cm.delete_to_trash = True
    sink = event_sink(cm)
    path = await make_file(cm)

    def fake_send2trash(os_path):
        raise OSError("trash unavailable")

    monkeypatch.setattr(
        "jupyter_server.services.contents.filemanager.send2trash", fake_send2trash
    )

    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.delete(path))
    assert exc.value.status_code == 400

    # Nothing was deleted: both content and checkpoint are still present.
    assert await ensure_async(cm.file_exists(path))
    assert len(await ensure_async(cm.list_checkpoints(path))) == 1
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path)) == []
    assert delete_events(sink) == []

    # Healthy retry (now permanent) completes the delete.
    monkeypatch.undo()
    cm.delete_to_trash = False
    await ensure_async(cm.delete(path))
    assert not await ensure_async(cm.file_exists(path))
    assert len(delete_events(sink)) == 1


# ---------------------------------------------------------------------------
# checkpoint cleanup (commit) fails after the content is already gone
# ---------------------------------------------------------------------------


async def test_checkpoint_commit_failure_is_retryable(permanent_delete, monkeypatch):
    cm = permanent_delete
    sink = event_sink(cm)
    path = await make_file(cm)
    real_purge = type(cm.checkpoints).purge_checkpoint_file
    calls = {"n": 0}

    if is_async(cm):

        async def flaky_purge(staged_path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("checkpoint storage suddenly unavailable")
            return await real_purge(cm.checkpoints, staged_path)

    else:

        def flaky_purge(staged_path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("checkpoint storage suddenly unavailable")
            return real_purge(cm.checkpoints, staged_path)

    monkeypatch.setattr(cm.checkpoints, "purge_checkpoint_file", flaky_purge)

    # First request: content gone, checkpoint commit failed -> 500 and no
    # "delete" event may be recorded.
    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.delete(path))
    assert exc.value.status_code == 500

    assert not await ensure_async(cm.file_exists(path))
    # The old checkpoint is hidden ...
    assert await ensure_async(cm.list_checkpoints(path)) == []
    # ... and the interrupted delete is recorded for recovery.
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path))
    assert delete_events(sink) == []

    # Retry finishes the interrupted delete and reports the real outcome.
    await ensure_async(cm.delete(path))
    assert await ensure_async(cm.list_checkpoints(path)) == []
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path)) == []
    assert len(delete_events(sink)) == 1


# ---------------------------------------------------------------------------
# content deletion fails before the content is removed
# ---------------------------------------------------------------------------


async def test_content_delete_failure_restores_checkpoints(permanent_delete, monkeypatch):
    cm = permanent_delete
    sink = event_sink(cm)
    path = await make_file(cm)
    fail_before_remove(cm, monkeypatch)

    with pytest.raises(RuntimeError):
        await ensure_async(cm.delete(path))

    # Both facts are back to the pre-delete state.
    assert await ensure_async(cm.file_exists(path))
    assert len(await ensure_async(cm.list_checkpoints(path))) == 1
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path)) == []
    assert delete_events(sink) == []

    # A later, healthy delete works normally and emits exactly one event.
    monkeypatch.undo()
    await ensure_async(cm.delete(path))
    assert not await ensure_async(cm.file_exists(path))
    assert len(delete_events(sink)) == 1


async def test_content_gone_then_delete_raises_is_recoverable(
    permanent_delete, monkeypatch
):
    """The reported bug: the file vanishes, then storage raises.

    The first request returns failure but leaves the checkpoint hidden and
    records an in-flight transaction.  A retry detects that the content is
    already gone, finishes the cleanup and reports a single, truthful
    completion -- instead of resurrecting the checkpoint for missing
    content.
    """
    cm = permanent_delete
    sink = event_sink(cm)
    path = await make_file(cm)
    fail_after_unlink(cm, path, monkeypatch)

    # First request: content gone but the backend then errored -> failure.
    with pytest.raises(RuntimeError):
        await ensure_async(cm.delete(path))

    assert not await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.list_checkpoints(path)) == []
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path))
    assert delete_events(sink) == []

    # Retry (resume branch; delete_file is never called for missing content)
    # completes the interrupted delete and now emits the event.
    await ensure_async(cm.delete(path))
    assert await ensure_async(cm.list_checkpoints(path)) == []
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path)) == []
    assert len(delete_events(sink)) == 1


# ---------------------------------------------------------------------------
# compensation failure
# ---------------------------------------------------------------------------


async def test_compensation_failure_is_healed_on_retry(permanent_delete, monkeypatch):
    cm = permanent_delete
    sink = event_sink(cm)
    path = await make_file(cm)
    real_restore = type(cm.checkpoints).restore_checkpoint_file
    calls = {"n": 0}

    if is_async(cm):

        async def flaky_restore(staged_path, checkpoint_path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("cannot restore checkpoint")
            return await real_restore(cm.checkpoints, staged_path, checkpoint_path)

    else:

        def flaky_restore(staged_path, checkpoint_path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("cannot restore checkpoint")
            return real_restore(cm.checkpoints, staged_path, checkpoint_path)

    monkeypatch.setattr(cm.checkpoints, "restore_checkpoint_file", flaky_restore)
    fail_before_remove(cm, monkeypatch)

    # First attempt: delete fails *and* the checkpoint cannot be restored.
    with pytest.raises(RuntimeError):
        await ensure_async(cm.delete(path))
    # The content survived; the orphaned transaction is retained for recovery.
    assert await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path))

    # Second attempt: compensation works now, the transaction is adopted and
    # healed, and the delete completes.
    monkeypatch.undo()
    await ensure_async(cm.delete(path))
    assert not await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.list_checkpoints(path)) == []
    assert await ensure_async(cm.checkpoints.find_delete_transactions(path)) == []
    assert len(delete_events(sink)) == 1


# ---------------------------------------------------------------------------
# duplicate requests
# ---------------------------------------------------------------------------


async def test_concurrent_duplicate_requests_async(permanent_delete, monkeypatch):
    cm = permanent_delete
    if not is_async(cm):
        pytest.skip("async-only concurrency test")
    sink = event_sink(cm)
    path = await make_file(cm)

    entered = asyncio.Event()
    release = asyncio.Event()
    real_delete_file = type(cm).delete_file

    async def gated_delete_file(p):
        entered.set()
        await release.wait()
        return await real_delete_file(cm, p)

    monkeypatch.setattr(cm, "delete_file", gated_delete_file)

    first = asyncio.ensure_future(cm.delete(path))
    await entered.wait()
    second = asyncio.ensure_future(cm.delete(path))  # genuine duplicate
    await asyncio.sleep(0.05)
    release.set()

    await asyncio.gather(first, second)  # neither raises

    assert not await ensure_async(cm.file_exists(path))
    assert len(delete_events(sink)) == 1


def test_concurrent_duplicate_requests_sync(tmp_path, monkeypatch):
    cm = FileContentsManager(root_dir=str(tmp_path))
    cm.delete_to_trash = False
    sink = event_sink(cm)
    cm.save({"type": "file", "format": "text", "content": "data"}, "f.txt")

    entered = threading.Event()
    release = threading.Event()
    real_delete_file = FileContentsManager.delete_file

    def gated_delete_file(p):
        entered.set()
        release.wait(timeout=5)
        return real_delete_file(cm, p)

    monkeypatch.setattr(cm, "delete_file", gated_delete_file)

    errors = []

    def do_delete():
        try:
            cm.delete("f.txt")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=do_delete)
    t1.start()
    entered.wait(5)
    t2 = threading.Thread(target=do_delete)  # overlaps the in-flight delete
    t2.start()
    release.set()
    t1.join(5)
    t2.join(5)

    assert errors == []
    assert not cm.file_exists("f.txt")
    assert len(delete_events(sink)) == 1


async def test_sequential_request_after_completion_is_404(permanent_delete):
    """A fresh request to delete an already-missing path keeps the 404."""
    cm = permanent_delete
    path = await make_file(cm)

    await ensure_async(cm.delete(path))
    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.delete(path))
    assert exc.value.status_code == 404


def test_concurrent_duplicate_across_manager_instances(tmp_path, monkeypatch):
    """Two managers sharing the root serialize via the cross-process claim."""
    cm1 = FileContentsManager(root_dir=str(tmp_path))
    cm2 = FileContentsManager(root_dir=str(tmp_path))
    cm1.delete_to_trash = False
    cm2.delete_to_trash = False
    sink1 = event_sink(cm1)
    sink2 = event_sink(cm2)
    cm1.save({"type": "file", "format": "text", "content": "data"}, "g.txt")

    entered = threading.Event()
    release = threading.Event()
    real_delete_file = FileContentsManager.delete_file

    def gated(p):
        entered.set()
        release.wait(timeout=5)
        return real_delete_file(cm1, p)

    monkeypatch.setattr(cm1, "delete_file", gated)

    errors = []

    def second():
        # Only start trying once the first delete holds the claim.
        entered.wait(5)
        _try(cm2, errors)

    t1 = threading.Thread(target=lambda: _try(cm1, errors))
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    release.set()
    t1.join(5)
    t2.join(5)

    assert errors == []
    assert not cm2.file_exists("g.txt")
    # Exactly one delete really happened, so exactly one event in total.
    assert len(delete_events(sink1)) + len(delete_events(sink2)) == 1


def _try(cm, errors):
    try:
        cm.delete("g.txt")
    except Exception as e:
        errors.append(e)


# ---------------------------------------------------------------------------
# legacy checkpoint stores without the staging protocol keep working
# ---------------------------------------------------------------------------


class _LegacySyncCheckpoints(FileCheckpoints):
    supports_delete_transactions = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deleted_all_for = []

    def delete_all_checkpoints(self, path):
        self.deleted_all_for.append(path)


class _LegacyAsyncCheckpoints(AsyncFileCheckpoints):
    supports_delete_transactions = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deleted_all_for = []

    async def delete_all_checkpoints(self, path):
        self.deleted_all_for.append(path)


async def test_legacy_checkpoints_store_uses_best_effort_delete(
    permanent_delete,
):
    cm = permanent_delete
    path = await make_file(cm)
    legacy_cls = (
        _LegacyAsyncCheckpoints if is_async(cm) else _LegacySyncCheckpoints
    )
    legacy = legacy_cls(parent=cm)
    cm.checkpoints = legacy
    sink = event_sink(cm)

    await ensure_async(cm.delete(path))

    assert not await ensure_async(cm.file_exists(path))
    assert legacy.deleted_all_for == [path]
    assert len(delete_events(sink)) == 1

    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.delete(path))
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# events describe only what really happened
# ---------------------------------------------------------------------------


async def test_no_event_on_validation_failure(permanent_delete):
    cm = permanent_delete
    sink = event_sink(cm)
    with pytest.raises(HTTPError):
        await ensure_async(cm.delete(""))
    assert delete_events(sink) == []


# ---------------------------------------------------------------------------
# rename keeps checkpoints consistent when the content move fails
# ---------------------------------------------------------------------------


async def test_rename_failure_moves_checkpoint_back(permanent_delete, monkeypatch):
    cm = permanent_delete
    sink = event_sink(cm)
    path = await make_notebook(cm, "r.ipynb")

    if is_async(cm):

        async def fail_rename(old, new):
            raise RuntimeError("move failed")

    else:

        def fail_rename(old, new):
            raise RuntimeError("move failed")

    monkeypatch.setattr(cm, "rename_file", fail_rename)

    with pytest.raises(RuntimeError):
        await ensure_async(cm.rename("r.ipynb", "s.ipynb"))

    assert await ensure_async(cm.file_exists("r.ipynb"))
    assert len(await ensure_async(cm.list_checkpoints("r.ipynb"))) == 1
    assert await ensure_async(cm.list_checkpoints("s.ipynb")) == []
    assert [e for e in recorded_events(sink) if e.get("action") == "rename"] == []
