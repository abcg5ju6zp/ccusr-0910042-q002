"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import os
import tempfile
import uuid
from functools import partial

from anyio.to_thread import run_sync
from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable


def _pending_root():
    """Root directory that holds in-flight delete staging areas."""
    return os.path.join(tempfile.gettempdir(), "jupyter_pending_checkpoint_deletes")


def _transaction_dir(txn_id):
    return os.path.join(_pending_root(), txn_id)


def _manifest_path(txn_id):
    return os.path.join(_transaction_dir(txn_id), "manifest.json")


def _write_manifest_atomic(path, root, path_api, entries):
    """Persist the transaction manifest via a temp file + rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"version": 1, "root": root, "path": path_api, "entries": entries}, f
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    entries = data.get("entries")
    if not isinstance(entries, list):
        return None
    return data


class Checkpoints(LoggingConfigurable):
    """项目内部接口说明。"""

    #: Whether this store implements the recoverable staging primitives
    #: (list_checkpoint_files/trash_/restore_/purge_checkpoint_file).
    #: Stores that keep ``False`` get the legacy best-effort delete
    #: ordering; :class:`FileCheckpoints` and friends opt in.
    supports_delete_transactions = False

    def create_checkpoint(self, contents_mgr, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def list_checkpoints(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def rename_all_checkpoints(self, old_path, new_path):
        """项目内部接口说明。"""
        for cp in self.list_checkpoints(old_path):
            self.rename_checkpoint(cp["id"], old_path, new_path)

    def delete_all_checkpoints(self, path):
        """项目内部接口说明。"""
        for checkpoint in self.list_checkpoints(path):
            self.delete_checkpoint(checkpoint["id"], path)

    # ------------------------------------------------------------------
    # Transactional checkpoint deletion
    #
    # ``ContentsManager.delete`` removes two independent resources -- the
    # content (file/directory) and its checkpoints -- which cannot be
    # removed atomically on a plain filesystem.  The protocol here keeps
    # the two facts consistent:
    #
    # 1. ``stage_checkpoint_deletes`` moves every checkpoint artifact into a
    #    private, manifest-described staging directory *before* the content
    #    is touched.  A failure (in staging or later in the contents
    #    manager) is repaired with ``rollback_checkpoint_deletes``.
    # 2. Once the content is gone for good, ``commit_checkpoint_deletes``
    #    purges the staged artifacts.
    #
    # The manifest records the API path being deleted and every moved
    # artifact, so an in-flight transaction is discoverable by any request
    # or process (``find_delete_transactions``): duplicate requests join or
    # adopt the existing transaction instead of racing, and a crashed
    # transaction can be resumed (content gone -> purge) or rolled back
    # (content present -> restore).
    #
    # Subclasses implement four small primitives; the default
    # implementations raise NotImplementedError like the rest of the API.
    # ------------------------------------------------------------------

    def list_checkpoint_files(self, path):
        """Return ``[(token, checkpoint_path), ...]`` for *path*.

        Reports every on-disk checkpoint artifact for the content, not only
        those that parse into a checkpoint model.  *token* is an opaque
        identifier unique within the result set.
        """
        raise NotImplementedError

    def trash_checkpoint_file(self, checkpoint_path, dest_path):
        """Move one checkpoint artifact to *dest_path* in staging.

        Should return ``False`` (without raising) when the source no longer
        exists; return ``True`` when the artifact was moved.
        """
        raise NotImplementedError

    def restore_checkpoint_file(self, staged_path, checkpoint_path):
        """Move a staged artifact back to its original location."""
        raise NotImplementedError

    def purge_checkpoint_file(self, staged_path):
        """Permanently remove a staged checkpoint artifact."""
        raise NotImplementedError

    # -- transaction orchestration ------------------------------------------

    def _root_dir(self):
        return getattr(self, "root_dir", None) or os.getcwd()

    def stage_checkpoint_deletes(self, path):
        """Move all checkpoints of *path* into a recoverable staging area.

        Returns ``(txn_id, entries)`` with entries of the shape
        ``{"token", "original", "staged"}``.  Either every checkpoint is
        staged, or anything already moved is rolled back and the original
        exception is re-raised.
        """
        txn_id = uuid.uuid4().hex
        txn_dir = _transaction_dir(txn_id)
        manifest = _manifest_path(txn_id)
        entries = []
        # Record the transaction (even when there are no checkpoints yet) so
        # a crash between this point and the commit is discoverable by a
        # retry.  The directory is removed again on rollback/commit.
        os.makedirs(txn_dir, exist_ok=True)
        _write_manifest_atomic(manifest, os.path.abspath(self._root_dir()), path, entries)
        try:
            for index, (token, checkpoint_path) in enumerate(
                self.list_checkpoint_files(path)
            ):
                if not os.path.lexists(checkpoint_path):
                    continue
                dest_path = os.path.join(txn_dir, f"{index:04d}")
                moved = self.trash_checkpoint_file(checkpoint_path, dest_path)
                if moved is False:
                    continue
                if not os.path.lexists(dest_path):
                    raise OSError(
                        "staging checkpoint %s did not produce %s"
                        % (checkpoint_path, dest_path)
                    )
                entries.append(
                    {"token": token, "original": checkpoint_path, "staged": dest_path}
                )
                # Persist progress so a crash mid-stage can be recovered.
                _write_manifest_atomic(
                    manifest, os.path.abspath(self._root_dir()), path, entries
                )
        except Exception:
            self.rollback_checkpoint_deletes(txn_id, entries)
            self._discard_transaction(txn_id)
            raise
        return txn_id, entries

    def rollback_checkpoint_deletes(self, txn_id, entries=None):
        """Restore staged checkpoints after an aborted delete.

        Returns the entries that could not be restored (empty means the
        original checkpoint state was fully recovered).
        """
        if entries is None:
            data = _read_manifest(_manifest_path(txn_id))
            entries = data["entries"] if data else []
        failed = []
        for entry in entries:
            staged_path = entry["staged"]
            original_path = entry["original"]
            if not os.path.lexists(staged_path):
                continue
            if os.path.lexists(original_path):
                # Do not clobber a checkpoint another request put back.
                continue
            try:
                self.restore_checkpoint_file(staged_path, original_path)
            except Exception:
                self.log.error(
                    "Failed to restore checkpoint %s after aborted delete",
                    original_path,
                    exc_info=True,
                )
                failed.append(entry)
        if not failed:
            self._discard_transaction(txn_id)
        return failed

    def commit_checkpoint_deletes(self, txn_id, entries=None):
        """Permanently remove the staged checkpoints of transaction *txn_id*."""
        if entries is None:
            data = _read_manifest(_manifest_path(txn_id))
            entries = data["entries"] if data else []
        for entry in entries:
            staged_path = entry["staged"]
            if not os.path.lexists(staged_path):
                continue
            try:
                self.purge_checkpoint_file(staged_path)
            except FileNotFoundError:
                pass
        self._discard_transaction(txn_id)

    def resume_checkpoint_deletes(self, path, txn_id):
        """Close a transaction whose content is already gone.

        Purges everything still staged for *path* and sweeps any checkpoint
        files that survived on disk.  Returns True when fully closed.
        """
        data = _read_manifest(_manifest_path(txn_id))
        entries = data["entries"] if data else []
        ok = True
        for entry in entries:
            staged_path = entry["staged"]
            if os.path.lexists(staged_path):
                try:
                    self.purge_checkpoint_file(staged_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    self.log.error(
                        "Failed to purge staged checkpoint %s during resume",
                        staged_path,
                        exc_info=True,
                    )
                    ok = False
        # The content is gone: checkpoint files still on disk are stale.
        for _token, checkpoint_path in self.list_checkpoint_files(path):
            if os.path.lexists(checkpoint_path):
                try:
                    self.purge_checkpoint_file(checkpoint_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    self.log.error(
                        "Failed to purge stale checkpoint %s during resume",
                        checkpoint_path,
                        exc_info=True,
                    )
                    ok = False
        if ok:
            self._discard_transaction(txn_id)
        return ok

    def find_delete_transactions(self, path):
        """Return ``[(txn_id, entries), ...]`` staged for content *path*.

        A transaction matches when its manifest targets *path* in this
        checkpoints store's root directory.  Used to join/adopt
        transactions left by duplicate or crashed requests.
        """
        root = _pending_root()
        if not os.path.isdir(root):
            return []
        root_abs = os.path.abspath(self._root_dir())
        matches = []
        for txn_id in os.listdir(root):
            data = _read_manifest(_manifest_path(txn_id))
            if (
                data
                and data.get("path") == path
                and data.get("root") == root_abs
            ):
                matches.append((txn_id, data["entries"]))
        return matches

    def _discard_transaction(self, txn_id):
        txn_dir = _transaction_dir(txn_id)
        try:
            for name in os.listdir(txn_dir):
                os.unlink(os.path.join(txn_dir, name))
            os.rmdir(txn_dir)
        except FileNotFoundError:
            pass
        except OSError:
            # Remaining files (e.g. an unpurged checkpoint) keep the
            # transaction discoverable for a later resume/rollback.
            pass


class GenericCheckpointsMixin:
    """项目内部接口说明。"""

    def create_checkpoint(self, contents_mgr, path):
        model = contents_mgr.get(path, content=True)
        type_ = model["type"]
        if type_ == "notebook":
            return self.create_notebook_checkpoint(
                model["content"],
                path,
            )
        elif type_ == "file":
            return self.create_file_checkpoint(
                model["content"],
                model["format"],
                path,
            )
        else:
            raise HTTPError(500, "Unexpected type %s" % type)

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        type_ = contents_mgr.get(path, content=False)["type"]
        if type_ == "notebook":
            model = self.get_notebook_checkpoint(checkpoint_id, path)
        elif type_ == "file":
            model = self.get_file_checkpoint(checkpoint_id, path)
        else:
            raise HTTPError(500, "Unexpected type %s" % type_)
        contents_mgr.save(model, path)

    # Required Methods
    def create_file_checkpoint(self, content, format, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def create_notebook_checkpoint(self, nb, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def get_file_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError


class AsyncCheckpoints(Checkpoints):
    """项目内部接口说明。"""

    async def create_checkpoint(self, contents_mgr, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def list_checkpoints(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def rename_all_checkpoints(self, old_path, new_path):
        """项目内部接口说明。"""
        for cp in await self.list_checkpoints(old_path):
            await self.rename_checkpoint(cp["id"], old_path, new_path)

    async def delete_all_checkpoints(self, path):
        """项目内部接口说明。"""
        for checkpoint in await self.list_checkpoints(path):
            await self.delete_checkpoint(checkpoint["id"], path)

    # ------------------------------------------------------------------
    # Async counterparts of the transactional primitives.
    # ------------------------------------------------------------------

    async def list_checkpoint_files(self, path):
        """Async counterpart of :meth:`Checkpoints.list_checkpoint_files`."""
        raise NotImplementedError

    async def trash_checkpoint_file(self, checkpoint_path, dest_path):
        """Async counterpart of :meth:`Checkpoints.trash_checkpoint_file`."""
        raise NotImplementedError

    async def restore_checkpoint_file(self, staged_path, checkpoint_path):
        """Async counterpart of :meth:`Checkpoints.restore_checkpoint_file`."""
        raise NotImplementedError

    async def purge_checkpoint_file(self, staged_path):
        """Async counterpart of :meth:`Checkpoints.purge_checkpoint_file`."""
        raise NotImplementedError

    async def stage_checkpoint_deletes(self, path):
        """项目内部接口说明。"""
        txn_id = uuid.uuid4().hex
        txn_dir = _transaction_dir(txn_id)
        manifest = _manifest_path(txn_id)
        entries = []
        await run_sync(partial(os.makedirs, exist_ok=True), txn_dir)
        await run_sync(
            _write_manifest_atomic,
            manifest,
            os.path.abspath(self._root_dir()),
            path,
            entries,
        )
        try:
            checkpoint_files = await self.list_checkpoint_files(path)
            for index, (token, checkpoint_path) in enumerate(checkpoint_files):
                if not os.path.lexists(checkpoint_path):
                    continue
                dest_path = os.path.join(txn_dir, f"{index:04d}")
                moved = await self.trash_checkpoint_file(checkpoint_path, dest_path)
                if moved is False:
                    continue
                if not os.path.lexists(dest_path):
                    raise OSError(
                        "staging checkpoint %s did not produce %s"
                        % (checkpoint_path, dest_path)
                    )
                entries.append(
                    {"token": token, "original": checkpoint_path, "staged": dest_path}
                )
                await run_sync(
                    _write_manifest_atomic,
                    manifest,
                    os.path.abspath(self._root_dir()),
                    path,
                    entries,
                )
        except Exception:
            await self.rollback_checkpoint_deletes(txn_id, entries)
            self._discard_transaction(txn_id)
            raise
        return txn_id, entries

    async def rollback_checkpoint_deletes(self, txn_id, entries=None):
        """项目内部接口说明。"""
        if entries is None:
            data = _read_manifest(_manifest_path(txn_id))
            entries = data["entries"] if data else []
        failed = []
        for entry in entries:
            staged_path = entry["staged"]
            original_path = entry["original"]
            if not os.path.lexists(staged_path):
                continue
            if os.path.lexists(original_path):
                continue
            try:
                await self.restore_checkpoint_file(staged_path, original_path)
            except Exception:
                self.log.error(
                    "Failed to restore checkpoint %s after aborted delete",
                    original_path,
                    exc_info=True,
                )
                failed.append(entry)
        if not failed:
            self._discard_transaction(txn_id)
        return failed

    async def commit_checkpoint_deletes(self, txn_id, entries=None):
        """项目内部接口说明。"""
        if entries is None:
            data = _read_manifest(_manifest_path(txn_id))
            entries = data["entries"] if data else []
        for entry in entries:
            staged_path = entry["staged"]
            if not os.path.lexists(staged_path):
                continue
            try:
                await self.purge_checkpoint_file(staged_path)
            except FileNotFoundError:
                pass
        self._discard_transaction(txn_id)

    async def resume_checkpoint_deletes(self, path, txn_id):
        """项目内部接口说明。"""
        data = _read_manifest(_manifest_path(txn_id))
        entries = data["entries"] if data else []
        ok = True
        for entry in entries:
            staged_path = entry["staged"]
            if os.path.lexists(staged_path):
                try:
                    await self.purge_checkpoint_file(staged_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    self.log.error(
                        "Failed to purge staged checkpoint %s during resume",
                        staged_path,
                        exc_info=True,
                    )
                    ok = False
        for _token, checkpoint_path in await self.list_checkpoint_files(path):
            if os.path.lexists(checkpoint_path):
                try:
                    await self.purge_checkpoint_file(checkpoint_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    self.log.error(
                        "Failed to purge stale checkpoint %s during resume",
                        checkpoint_path,
                        exc_info=True,
                    )
                    ok = False
        if ok:
            self._discard_transaction(txn_id)
        return ok

    async def find_delete_transactions(self, path):
        """项目内部接口说明。"""
        root = _pending_root()
        if not os.path.isdir(root):
            return []
        root_abs = os.path.abspath(self._root_dir())
        matches = []
        for txn_id in await run_sync(os.listdir, root):
            data = _read_manifest(_manifest_path(txn_id))
            if (
                data
                and data.get("path") == path
                and data.get("root") == root_abs
            ):
                matches.append((txn_id, data["entries"]))
        return matches


class AsyncGenericCheckpointsMixin(GenericCheckpointsMixin):
    """项目内部接口说明。"""

    async def create_checkpoint(self, contents_mgr, path):
        model = await contents_mgr.get(path, content=True)
        type_ = model["type"]
        if type_ == "notebook":
            return await self.create_notebook_checkpoint(
                model["content"],
                path,
            )
        elif type_ == "file":
            return await self.create_file_checkpoint(
                model["content"],
                model["format"],
                path,
            )
        else:
            raise HTTPError(500, "Unexpected type %s" % type_)

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        content_model = await contents_mgr.get(path, content=False)
        type_ = content_model["type"]
        if type_ == "notebook":
            model = await self.get_notebook_checkpoint(checkpoint_id, path)
        elif type_ == "file":
            model = await self.get_file_checkpoint(checkpoint_id, path)
        else:
            raise HTTPError(500, "Unexpected type %s" % type_)
        await contents_mgr.save(model, path)

    # Required Methods
    async def create_file_checkpoint(self, content, format, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def create_notebook_checkpoint(self, nb, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def get_file_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def get_notebook_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError
