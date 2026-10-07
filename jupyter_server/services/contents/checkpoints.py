"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable


class Checkpoints(LoggingConfigurable):
    """项目内部接口说明。"""

    # Whether this backend can move checkpoint files into a delete
    # transaction staging area and restore/discard them afterwards.
    supports_staged_purge = False

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

    def purge_checkpoints(self, path):
        """Remove every checkpoint associated with ``path``.

        Unlike :meth:`delete_all_checkpoints`, this is intended for content
        deletion and must be safe to call when the content is already gone
        (e.g. retries of an interrupted delete).

        Returns the number of checkpoint records that were removed. A return
        value of ``0`` means there was nothing to clean up.

        The purge is idempotent: a checkpoint that disappeared concurrently
        (e.g. deleted by a duplicate request) is ignored rather than raised.
        If removal fails midway, a subsequent call resumes from whatever
        remains so the store converges across retries.
        """
        removed = 0
        for checkpoint in self.list_checkpoints(path):
            try:
                self.delete_checkpoint(checkpoint["id"], path)
            except HTTPError as e:
                # Deleted by a concurrent request; not a failure of the purge.
                if e.status_code != 404:
                    raise
                continue
            removed += 1
        return removed

    def has_pending_checkpoints(self, path):
        """Whether any checkpoint record for ``path`` still exists."""
        return bool(self.list_checkpoints(path))

    # Recoverable-purge primitives. The default checkpoint backend (files on
    # disk) implements these; alternative backends paired with a
    # transaction-capable ContentsManager should implement them too. A backend
    # without staging support can keep using :meth:`purge_checkpoints` with
    # the manager's non-transactional delete ordering.
    def stage_purge_checkpoints(self, path, staging_dir, extra_paths=None):
        """Move checkpoint files aside for a delete transaction."""
        raise NotImplementedError

    def fallback_checkpoint_files(self, path):
        """Checkpoint files stored outside the regular location for ``path``."""
        return []

    def restore_staged_checkpoints(self, moves):
        """Move previously staged checkpoint files back to their origin."""
        raise NotImplementedError

    def discard_staged_checkpoints(self, moves, *, remove_originals=True):
        """Permanently discard previously staged checkpoint files."""
        raise NotImplementedError


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

    async def purge_checkpoints(self, path):
        """Async variant of :meth:`Checkpoints.purge_checkpoints`.

        Idempotent: missing checkpoints (404) are treated as already purged.
        Returns the number of checkpoint records removed.
        """
        removed = 0
        for checkpoint in await self.list_checkpoints(path):
            try:
                await self.delete_checkpoint(checkpoint["id"], path)
            except HTTPError as e:
                # Deleted by a concurrent request; not a failure of the purge.
                if e.status_code != 404:
                    raise
                continue
            removed += 1
        return removed

    async def has_pending_checkpoints(self, path):
        """Whether any checkpoint record for ``path`` still exists."""
        return bool(await self.list_checkpoints(path))


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
