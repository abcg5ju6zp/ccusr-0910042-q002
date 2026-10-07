"""项目内部接口说明。"""

from __future__ import annotations

import os
import shutil
import tempfile

from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_dir_exists
from tornado.web import HTTPError
from traitlets import Unicode

from jupyter_server import _tz as tz

from .checkpoints import (
    AsyncCheckpoints,
    AsyncGenericCheckpointsMixin,
    Checkpoints,
    GenericCheckpointsMixin,
)
from .fileio import AsyncFileManagerMixin, FileManagerMixin


class FileCheckpoints(FileManagerMixin, Checkpoints):
    """项目内部接口说明。"""

    supports_staged_purge = True

    checkpoint_dir = Unicode(
        ".ipynb_checkpoints",
        config=True,
        help="""The directory name in which to keep file checkpoints

        This is a path relative to the file's own directory.

        By default, it is .ipynb_checkpoints
        """,
    )

    root_dir = Unicode(config=True)

    def _root_dir_default(self):
        if not self.parent:
            return os.getcwd()
        return self.parent.root_dir

    # ContentsManager-dependent checkpoint API
    def create_checkpoint(self, contents_mgr, path):
        """项目内部接口说明。"""
        checkpoint_id = "checkpoint"
        src_path = contents_mgr._get_os_path(path)
        dest_path = self.checkpoint_path(checkpoint_id, path)
        self._copy(src_path, dest_path)
        return self.checkpoint_model(checkpoint_id, dest_path)

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        src_path = self.checkpoint_path(checkpoint_id, path)
        dest_path = contents_mgr._get_os_path(path)
        self._copy(src_path, dest_path)

    # ContentsManager-independent checkpoint API
    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """项目内部接口说明。"""
        old_cp_path = self.checkpoint_path(checkpoint_id, old_path)
        new_cp_path = self.checkpoint_path(checkpoint_id, new_path)
        if os.path.isfile(old_cp_path):
            self.log.debug(
                "Renaming checkpoint %s -> %s",
                old_cp_path,
                new_cp_path,
            )
            with self.perm_to_403():
                shutil.move(old_cp_path, new_cp_path)

    def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        cp_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(cp_path):
            self.no_such_checkpoint(path, checkpoint_id)

        self.log.debug("unlinking %s", cp_path)
        with self.perm_to_403():
            os.unlink(cp_path)

    def list_checkpoints(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        checkpoint_id = "checkpoint"
        os_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(os_path):
            return []
        else:
            return [self.checkpoint_model(checkpoint_id, os_path)]

    # Recoverable checkpoint purge used while deleting content. The contents
    # manager owns the transaction directory (same filesystem as the content
    # whenever possible); these helpers only move checkpoint files in and out
    # of it, so an aborted delete can restore every checkpoint record. The
    # helpers are plain filesystem operations and never call the (possibly
    # async) checkpoint API, so the async manager can run them in a thread.
    def stage_purge_checkpoints(self, path, staging_dir, extra_paths=None):
        """Move checkpoint files of ``path`` under ``staging_dir``.

        ``extra_paths`` lets the caller add checkpoint files stored outside
        the regular sibling location (e.g. records kept in the temp fallback
        tree for contents of a deleted directory).

        Returns a list of ``(original_path, staged_path)`` pairs.
        """
        path = path.strip("/")
        cp_root = os.path.join(staging_dir, "checkpoints")
        candidates = [self._primary_checkpoint_path(path)]
        if extra_paths:
            candidates.extend(extra_paths)

        moves: list[tuple[str, str]] = []
        for index, src in enumerate(dict.fromkeys(candidates)):
            if not os.path.isfile(src):
                continue
            dest = os.path.join(cp_root, f"{index}-{os.path.basename(src)}")
            ensure_dir_exists(os.path.dirname(dest))
            try:
                with self.perm_to_403():
                    shutil.move(src, dest)
            except Exception:
                # Undo the moves already made so a failed staging leaves the
                # checkpoint store in its original state.
                self.restore_staged_checkpoints(moves)
                raise
            moves.append((src, dest))
        return moves

    def restore_staged_checkpoints(self, moves: list[tuple[str, str]]) -> None:
        """Move previously staged checkpoint files back to their origin."""
        errors: list[OSError] = []
        for original, staged in reversed(moves):
            if not os.path.isfile(staged):
                errors.append(FileNotFoundError(staged))
                continue
            try:
                ensure_dir_exists(os.path.dirname(original))
                shutil.move(staged, original)
            except OSError as e:
                errors.append(e)
                self.log.error("Failed to restore checkpoint %s from %s: %s", original, staged, e)
        if errors:
            raise errors[0]

    def fallback_checkpoint_files(self, path):
        """Checkpoint files stored in the temp fallback tree for ``path``.

        When a directory is deleted, checkpoints for files nested inside it may
        live under the system temp directory because their original folders
        were read-only. The fallback tree is owned by the checkpoint store, so
        every file mirrored below ``path`` is a checkpoint record.
        """
        path = path.strip("/")
        mirror = os.path.join(tempfile.gettempdir(), "jupyter_checkpoints", path)
        if not path or not os.path.isdir(mirror):
            return []
        result: list[str] = []
        for root, _dirs, files in os.walk(mirror):
            result.extend(os.path.join(root, name) for name in files)
        return result

    def discard_staged_checkpoints(self, moves, *, remove_originals=True):
        """Permanently unlink staged checkpoint files.

        By default both the staged copy and (if present) a copy that a
        previous compensation moved back to the original location are removed,
        which makes committing an interrupted transaction idempotent. When the
        content at the original path was recreated after a crash,
        ``remove_originals=False`` limits cleanup to the staged copies so the
        new content's checkpoints are left untouched. Missing files are
        tolerated; filesystem errors propagate so the caller can retry.
        """
        targets = [staged for _original, staged in moves]
        if remove_originals:
            targets += [original for original, _staged in moves]
        for target in targets:
            if os.path.isfile(target):
                with self.perm_to_403():
                    os.unlink(target)

    # Checkpoint-related utilities
    def _primary_checkpoint_path(self, path):
        """Path of the canonical checkpoint record without creating dirs."""
        path = path.strip("/")
        parent, name = ("/" + path).rsplit("/", 1)
        parent = parent.strip("/")
        basename, ext = os.path.splitext(name)
        filename = f"{basename}-checkpoint{ext}"
        os_path = self._get_os_path(path=parent)
        cp_dir = os.path.join(os_path, self.checkpoint_dir)
        # Mirror the fallback used by checkpoint_path(): read-only parents
        # keep checkpoints under the system temp directory.
        if not os.access(os.path.dirname(cp_dir), os.W_OK):
            rel = os.path.relpath(os_path, start=self.root_dir)
            cp_dir = os.path.join(tempfile.gettempdir(), "jupyter_checkpoints", rel)
        return os.path.join(cp_dir, filename)

    def checkpoint_path(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        parent, name = ("/" + path).rsplit("/", 1)
        parent = parent.strip("/")
        basename, ext = os.path.splitext(name)
        filename = f"{basename}-{checkpoint_id}{ext}"
        os_path = self._get_os_path(path=parent)
        cp_dir = os.path.join(os_path, self.checkpoint_dir)
        # If parent directory isn't writable, use system temp
        if not os.access(os.path.dirname(cp_dir), os.W_OK):
            rel = os.path.relpath(os_path, start=self.root_dir)
            cp_dir = os.path.join(tempfile.gettempdir(), "jupyter_checkpoints", rel)
        with self.perm_to_403():
            ensure_dir_exists(cp_dir)
        cp_path = os.path.join(cp_dir, filename)
        return cp_path

    def checkpoint_model(self, checkpoint_id, os_path):
        """项目内部接口说明。"""
        stats = os.stat(os_path)
        last_modified = tz.utcfromtimestamp(stats.st_mtime)
        info = {
            "id": checkpoint_id,
            "last_modified": last_modified,
        }
        return info

    # Error Handling
    def no_such_checkpoint(self, path, checkpoint_id):
        raise HTTPError(404, f"Checkpoint does not exist: {path}@{checkpoint_id}")


class AsyncFileCheckpoints(FileCheckpoints, AsyncFileManagerMixin, AsyncCheckpoints):
    async def create_checkpoint(self, contents_mgr, path):
        """项目内部接口说明。"""
        checkpoint_id = "checkpoint"
        src_path = contents_mgr._get_os_path(path)
        dest_path = self.checkpoint_path(checkpoint_id, path)
        await self._copy(src_path, dest_path)
        return await self.checkpoint_model(checkpoint_id, dest_path)

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        src_path = self.checkpoint_path(checkpoint_id, path)
        dest_path = contents_mgr._get_os_path(path)
        await self._copy(src_path, dest_path)

    async def checkpoint_model(self, checkpoint_id, os_path):
        """项目内部接口说明。"""
        stats = await run_sync(os.stat, os_path)
        last_modified = tz.utcfromtimestamp(stats.st_mtime)
        info = {
            "id": checkpoint_id,
            "last_modified": last_modified,
        }
        return info

    # ContentsManager-independent checkpoint API
    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """项目内部接口说明。"""
        old_cp_path = self.checkpoint_path(checkpoint_id, old_path)
        new_cp_path = self.checkpoint_path(checkpoint_id, new_path)
        if os.path.isfile(old_cp_path):
            self.log.debug(
                "Renaming checkpoint %s -> %s",
                old_cp_path,
                new_cp_path,
            )
            with self.perm_to_403():
                await run_sync(shutil.move, old_cp_path, new_cp_path)

    async def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        cp_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(cp_path):
            self.no_such_checkpoint(path, checkpoint_id)

        self.log.debug("unlinking %s", cp_path)
        with self.perm_to_403():
            await run_sync(os.unlink, cp_path)

    async def list_checkpoints(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        checkpoint_id = "checkpoint"
        os_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(os_path):
            return []
        else:
            return [await self.checkpoint_model(checkpoint_id, os_path)]


class GenericFileCheckpoints(GenericCheckpointsMixin, FileCheckpoints):
    """项目内部接口说明。"""

    def create_file_checkpoint(self, content, format, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        # only the one checkpoint ID:
        checkpoint_id = "checkpoint"
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        with self.perm_to_403():
            self._save_file(os_checkpoint_path, content, format=format)

        # return the checkpoint info
        return self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    def create_notebook_checkpoint(self, nb, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        # only the one checkpoint ID:
        checkpoint_id = "checkpoint"
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        with self.perm_to_403():
            self._save_notebook(os_checkpoint_path, nb)

        # return the checkpoint info
        return self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)

        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)

        return {
            "type": "notebook",
            "content": self._read_notebook(
                os_checkpoint_path,
                as_version=4,
            ),
        }

    def get_file_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)

        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)

        content, format = self._read_file(os_checkpoint_path, format=None)  # type: ignore[misc]
        return {
            "type": "file",
            "content": content,
            "format": format,
        }


class AsyncGenericFileCheckpoints(AsyncGenericCheckpointsMixin, AsyncFileCheckpoints):
    """项目内部接口说明。"""

    async def create_file_checkpoint(self, content, format, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        # only the one checkpoint ID:
        checkpoint_id = "checkpoint"
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        with self.perm_to_403():
            await self._save_file(os_checkpoint_path, content, format=format)

        # return the checkpoint info
        return await self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    async def create_notebook_checkpoint(self, nb, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        # only the one checkpoint ID:
        checkpoint_id = "checkpoint"
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        with self.perm_to_403():
            await self._save_notebook(os_checkpoint_path, nb)

        # return the checkpoint info
        return await self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    async def get_notebook_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)

        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)

        return {
            "type": "notebook",
            "content": await self._read_notebook(
                os_checkpoint_path,
                as_version=4,
            ),
        }

    async def get_file_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)

        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)

        content, format = await self._read_file(os_checkpoint_path, format=None)  # type: ignore[misc]
        return {
            "type": "file",
            "content": content,
            "format": format,
        }
