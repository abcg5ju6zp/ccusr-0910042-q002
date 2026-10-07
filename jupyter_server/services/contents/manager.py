"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import threading
import typing as t
import warnings
from contextlib import contextmanager
from fnmatch import fnmatch

from jupyter_core.utils import ensure_async, run_sync
from jupyter_events import EventLogger
from nbformat import ValidationError, sign
from nbformat import validate as validate_nb
from nbformat.v4 import new_notebook
from tornado.web import HTTPError, RequestHandler
from traitlets import (
    Any,
    Bool,
    Dict,
    Instance,
    List,
    TraitError,
    Type,
    Unicode,
    default,
    validate,
)
from traitlets.config.configurable import LoggingConfigurable

from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
from jupyter_server.transutils import _i18n
from jupyter_server.utils import import_item

from ...files.handlers import FilesHandler
from .checkpoints import AsyncCheckpoints, Checkpoints

copy_pat = re.compile(r"\-Copy\d*\.")


class ContentsManager(LoggingConfigurable):
    """项目内部接口说明。"""

    event_schema_id = JUPYTER_SERVER_EVENTS_URI + "/contents_service/v1"
    event_logger = Instance(EventLogger).tag(config=True)

    @default("event_logger")
    def _default_event_logger(self):
        if self.parent and hasattr(self.parent, "event_logger"):
            return self.parent.event_logger
        else:
            # If parent does not have an event logger, create one.
            logger = EventLogger()
            schema_path = DEFAULT_EVENTS_SCHEMA_PATH / "contents_service" / "v1.yaml"
            logger.register_event_schema(schema_path)
            return logger

    def emit(self, data):
        """项目内部接口说明。"""
        self.event_logger.emit(schema_id=self.event_schema_id, data=data)

    root_dir = Unicode("/", config=True)

    preferred_dir = Unicode(
        "",
        config=True,
        help=_i18n(
            "Preferred starting directory to use for notebooks. This is an API path (`/` separated, relative to root dir)"
        ),
    )

    @validate("preferred_dir")
    def _validate_preferred_dir(self, proposal):
        value = proposal["value"].strip("/")
        try:
            import inspect

            if inspect.iscoroutinefunction(self.dir_exists):
                dir_exists = run_sync(self.dir_exists)(value)
            else:
                dir_exists = self.dir_exists(value)
        except HTTPError as e:
            raise TraitError(e.log_message) from e
        if not dir_exists:
            raise TraitError(_i18n("Preferred directory not found: %r") % value)
        if self.parent:
            try:
                if value != self.parent.preferred_dir:
                    self.parent.preferred_dir = os.path.join(self.root_dir, *value.split("/"))
            except TraitError:
                pass
        return value

    allow_hidden = Bool(False, config=True, help="Allow access to hidden files")

    notary = Instance(sign.NotebookNotary)

    @default("notary")
    def _notary_default(self):
        return sign.NotebookNotary(parent=self)

    hide_globs = List(
        Unicode(),
        [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*~",
        ],
        config=True,
        help="""
        Glob patterns to hide in file and directory listings.
    """,
    )

    untitled_notebook = Unicode(
        _i18n("Untitled"),
        config=True,
        help="The base name used when creating untitled notebooks.",
    )

    untitled_file = Unicode(
        "untitled", config=True, help="The base name used when creating untitled files."
    )

    untitled_directory = Unicode(
        "Untitled Folder",
        config=True,
        help="The base name used when creating untitled directories.",
    )

    pre_save_hook = Any(
        None,
        config=True,
        allow_none=True,
        help="""Python callable or importstring thereof

        To be called on a contents model prior to save.

        This can be used to process the structure,
        such as removing notebook outputs or other side effects that
        should not be saved.

        It will be called as (all arguments passed by keyword)::

            hook(path=path, model=model, contents_manager=self)

        - model: the model to be saved. Includes file contents.
          Modifying this dict will affect the file that is stored.
        - path: the API path of the save destination
        - contents_manager: this ContentsManager instance
        """,
    )

    @validate("pre_save_hook")
    def _validate_pre_save_hook(self, proposal):
        value = proposal["value"]
        if isinstance(value, str):
            value = import_item(self.pre_save_hook)
        if not callable(value):
            msg = "pre_save_hook must be callable"
            raise TraitError(msg)
        if callable(self.pre_save_hook):
            warnings.warn(
                f"Overriding existing pre_save_hook ({self.pre_save_hook.__name__}) with a new one ({value.__name__}).",
                stacklevel=2,
            )
        return value

    post_save_hook = Any(
        None,
        config=True,
        allow_none=True,
        help="""Python callable or importstring thereof

        to be called on the path of a file just saved.

        This can be used to process the file on disk,
        such as converting the notebook to a script or HTML via nbconvert.

        It will be called as (all arguments passed by keyword)::

            hook(os_path=os_path, model=model, contents_manager=instance)

        - path: the filesystem path to the file just written
        - model: the model representing the file
        - contents_manager: this ContentsManager instance
        """,
    )

    @validate("post_save_hook")
    def _validate_post_save_hook(self, proposal):
        value = proposal["value"]
        if isinstance(value, str):
            value = import_item(value)
        if not callable(value):
            msg = "post_save_hook must be callable"
            raise TraitError(msg)
        if callable(self.post_save_hook):
            warnings.warn(
                f"Overriding existing post_save_hook ({self.post_save_hook.__name__}) with a new one ({value.__name__}).",
                stacklevel=2,
            )
        return value

    def run_pre_save_hook(self, model, path, **kwargs):
        """项目内部接口说明。"""
        warnings.warn(
            "run_pre_save_hook is deprecated, use run_pre_save_hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.pre_save_hook:
            try:
                self.log.debug("Running pre-save hook on %s", path)
                self.pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
            except HTTPError:
                # allow custom HTTPErrors to raise,
                # rejecting the save with a message.
                raise
            except Exception:
                # unhandled errors don't prevent saving,
                # which could cause frustrating data loss
                self.log.error("Pre-save hook failed on %s", path, exc_info=True)

    def run_post_save_hook(self, model, os_path):
        """项目内部接口说明。"""
        warnings.warn(
            "run_post_save_hook is deprecated, use run_post_save_hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.post_save_hook:
            try:
                self.log.debug("Running post-save hook on %s", os_path)
                self.post_save_hook(os_path=os_path, model=model, contents_manager=self)
            except Exception:
                self.log.error("Post-save hook failed o-n %s", os_path, exc_info=True)
                msg = "fUnexpected error while running post hook save: {e}"
                raise HTTPError(500, msg) from None

    _pre_save_hooks: List[t.Any] = List()
    _post_save_hooks: List[t.Any] = List()

    def register_pre_save_hook(self, hook):
        """项目内部接口说明。"""
        if isinstance(hook, str):
            hook = import_item(hook)
        if not callable(hook):
            msg = "hook must be callable"
            raise RuntimeError(msg)
        self._pre_save_hooks.append(hook)

    def register_post_save_hook(self, hook):
        """项目内部接口说明。"""
        if isinstance(hook, str):
            hook = import_item(hook)
        if not callable(hook):
            msg = "hook must be callable"
            raise RuntimeError(msg)
        self._post_save_hooks.append(hook)

    def run_pre_save_hooks(self, model, path, **kwargs):
        """项目内部接口说明。"""
        pre_save_hooks = [self.pre_save_hook] if self.pre_save_hook is not None else []
        pre_save_hooks += self._pre_save_hooks
        for pre_save_hook in pre_save_hooks:
            try:
                self.log.debug("Running pre-save hook on %s", path)
                pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
            except HTTPError:
                # allow custom HTTPErrors to raise,
                # rejecting the save with a message.
                raise
            except Exception:
                # unhandled errors don't prevent saving,
                # which could cause frustrating data loss
                self.log.error(
                    "Pre-save hook %s failed on %s",
                    pre_save_hook.__name__,
                    path,
                    exc_info=True,
                )

    def run_post_save_hooks(self, model, os_path):
        """项目内部接口说明。"""
        post_save_hooks = [self.post_save_hook] if self.post_save_hook is not None else []
        post_save_hooks += self._post_save_hooks
        for post_save_hook in post_save_hooks:
            try:
                self.log.debug("Running post-save hook on %s", os_path)
                post_save_hook(os_path=os_path, model=model, contents_manager=self)
            except Exception as e:
                self.log.error(
                    "Post-save %s hook failed on %s",
                    post_save_hook.__name__,
                    os_path,
                    exc_info=True,
                )
                raise HTTPError(500, "Unexpected error while running post hook save: %s" % e) from e

    checkpoints_class = Type(Checkpoints, config=True)
    checkpoints = Instance(Checkpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default("checkpoints")
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default("checkpoints_kwargs")
    def _default_checkpoints_kwargs(self):
        return {
            "parent": self,
            "log": self.log,
        }

    files_handler_class = Type(
        FilesHandler,
        klass=RequestHandler,
        allow_none=True,
        config=True,
        help="""handler class to use when serving raw file requests.

        Default is a fallback that talks to the ContentsManager API,
        which may be inefficient, especially for large files.

        Local files-based ContentsManagers can use a StaticFileHandler subclass,
        which will be much more efficient.

        Access to these files should be Authenticated.
        """,
    )

    files_handler_params = Dict(
        config=True,
        help="""Extra parameters to pass to files_handler_class.

        For example, StaticFileHandlers generally expect a `path` argument
        specifying the root directory from which to serve files.
        """,
    )

    def get_extra_handlers(self):
        """项目内部接口说明。"""
        handlers = []
        if self.files_handler_class:
            handlers.append((r"/files/(.*)", self.files_handler_class, self.files_handler_params))
        return handlers

    # ContentsManager API part 1: methods that must be
    # implemented in subclasses.

    def dir_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def is_hidden(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def file_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def exists(self, path):
        """项目内部接口说明。"""
        return self.file_exists(path) or self.dir_exists(path)

    def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        raise NotImplementedError

    def save(self, model, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def delete_file(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    # ------------------------------------------------------------------
    # Delete serialization / recovery state
    #
    # ``delete`` touches two independent resources (content and
    # checkpoints).  Per-path locks serialize duplicate requests inside
    # this process; a file-based claim serializes other processes sharing
    # the root dir.  Whether a request had to wait for another delete
    # (``in_flight`` / claim contention) is what distinguishes a genuine
    # duplicate -- answered idempotently -- from a plain request to delete
    # something already missing -- answered with 404, as before.
    # ------------------------------------------------------------------

    def _delete_state(self):
        """Lazily initialize per-manager delete bookkeeping."""
        state = getattr(self, "_delete_bookkeeping", None)
        if state is None:
            state = {
                "locks": {},
                "locks_guard": threading.Lock(),
                "in_flight": set(),
            }
            self._delete_bookkeeping = state
        return state

    def _path_delete_lock(self, path):
        state = self._delete_state()
        with state["locks_guard"]:
            lock = state["locks"].get(path)
            if lock is None:
                lock = threading.Lock()
                state["locks"][path] = lock
            return lock

    def _delete_was_active(self, path):
        """Whether a delete for *path* was running when this request arrived."""
        state = self._delete_state()
        with state["locks_guard"]:
            return path in state["in_flight"]

    def _mark_delete_active(self, path, active):
        state = self._delete_state()
        with state["locks_guard"]:
            if active:
                state["in_flight"].add(path)
            else:
                state["in_flight"].discard(path)

    @contextmanager
    def _path_delete_claim(self, path):
        """Cross-process claim for deleting *path*.

        Yields ``contended`` via the returned context's result: True when
        another process held the claim and this request had to wait.  A
        claim held longer than the staleness threshold is treated as
        orphaned (its holder crashed) and stolen.
        """
        claim, contended = self._acquire_delete_claim(path)
        try:
            yield contended
        finally:
            self._release_delete_claim(claim)

    def _claim_path(self, path):
        import hashlib
        import tempfile

        claims_dir = os.path.join(
            tempfile.gettempdir(), "jupyter_pending_checkpoint_deletes", "claims"
        )
        os.makedirs(claims_dir, exist_ok=True)
        root = getattr(self, "root_dir", os.getcwd())
        token = hashlib.sha1(
            os.path.abspath(root).encode() + b"\0" + path.encode("utf-8")
        ).hexdigest()
        return os.path.join(claims_dir, token)

    def _acquire_delete_claim(self, path):
        """Create (or wait for / steal) the cross-process claim for *path*.

        Returns ``(claim_path, contended)``.
        """
        import time

        claim = self._claim_path(path)
        fd = None
        contended = False
        for _ in range(80):  # up to ~8 seconds
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError:
                contended = True
                try:
                    if time.time() - os.stat(claim).st_mtime > 30.0:
                        # Held longer than any delete should take: the
                        # holder probably crashed; steal the claim.
                        os.unlink(claim)
                        continue
                except FileNotFoundError:
                    pass
                time.sleep(0.1)
        if fd is None:
            raise HTTPError(409, "A delete for %r is already in progress" % path)
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
        finally:
            os.close(fd)
        # Refresh mtime so a live holder is never considered stale.
        os.utime(claim, None)
        return claim, contended

    def _release_delete_claim(self, claim):
        try:
            os.unlink(claim)
        except FileNotFoundError:
            pass

    def delete(self, path):
        """Delete content and its checkpoints as one recoverable operation.

        Ordering:

        1. Checkpoints are *staged* (moved to a private, manifest-backed
           area) first, so they stop being visible before the content can
           disappear -- never the other way round.
        2. The content is then removed (trash or permanent delete).
        3. Only once the content is actually gone are the staged
           checkpoints purged and the ``delete`` event emitted.

        Any failure after staging rolls the checkpoints back, leaving
        content and checkpoints as they were.  A failure after the content
        is gone leaves an on-disk transaction record, so a retry detects
        it, finishes the cleanup, and reports success.  A duplicate request
        that raced an in-flight delete is answered idempotently.  The
        ``delete`` event therefore describes a delete that truly completed.
        """
        path = path.strip("/")
        if not path:
            raise HTTPError(400, "Can't delete root")

        lock = self._path_delete_lock(path)
        duplicate = self._delete_was_active(path)
        with lock:
            self._mark_delete_active(path, True)
            try:
                with self._path_delete_claim(path) as claim_contended:
                    self._delete_unlocked(path, duplicate=duplicate or claim_contended)
            finally:
                self._mark_delete_active(path, False)

    def _resume_interrupted_delete(self, path, pending):
        """Finish a delete whose content is already gone.

        Returns ``None`` when there was no trace of an interrupted delete,
        ``True`` when the state was fully cleaned and ``False`` when cleanup
        is still incomplete (retryable).
        """
        did_work = False
        ok = True
        for txn_id, _entries in pending:
            did_work = True
            if not self.checkpoints.resume_checkpoint_deletes(path, txn_id):
                ok = False
        # Sweep stale checkpoint files not covered by a recorded transaction.
        for _token, checkpoint_path in self.checkpoints.list_checkpoint_files(path):
            if not os.path.lexists(checkpoint_path):
                continue
            did_work = True
            try:
                self.checkpoints.purge_checkpoint_file(checkpoint_path)
            except FileNotFoundError:
                pass
            except Exception:
                self.log.error(
                    "Failed to purge stale checkpoint %s while resuming delete of %r",
                    checkpoint_path,
                    path,
                    exc_info=True,
                )
                ok = False
        if not did_work:
            return None
        return ok

    def _delete_unlocked(self, path, duplicate=False):
        content_exists = self.exists(path)
        if not content_exists and duplicate:
            # Lost the race to a completed delete: report success once.
            return

        if not getattr(self.checkpoints, "supports_delete_transactions", False):
            # Legacy checkpoint stores: keep the historical best-effort
            # ordering without the staging/recovery protocol.
            if not content_exists:
                raise HTTPError(404, "file or directory does not exist: %r" % path)
            self.delete_file(path)
            self.checkpoints.delete_all_checkpoints(path)
            self.emit(data={"action": "delete", "path": path})
            return

        pending = self.checkpoints.find_delete_transactions(path)

        if not content_exists:
            resumed = self._resume_interrupted_delete(path, pending)
            if resumed is None:
                if duplicate:
                    # Lost the race to a fully completed delete: report the
                    # same success but do not record a second event.
                    return
                raise HTTPError(404, "file or directory does not exist: %r" % path)
            if not resumed:
                raise HTTPError(
                    500,
                    "Delete of %r was interrupted and could not be fully completed; "
                    "the content is gone but some checkpoints remain. Retry the request."
                    % path,
                )
            # The interrupted delete is only now truly finished (by this
            # request), so it is correct to emit the event here.
            self.emit(data={"action": "delete", "path": path})
            return

        # Content present: adopt orphaned transactions from earlier attempts
        # and restore their checkpoints before running a fresh delete.
        adopted_pending = []
        for txn_id, entries in pending:
            failed = self.checkpoints.rollback_checkpoint_deletes(txn_id, entries)
            adopted_pending.append(txn_id)
            if failed:
                self.log.error(
                    "Interrupted delete transaction %s for %r left %d checkpoint(s) "
                    "in the staging area; proceeding with a new delete",
                    txn_id,
                    path,
                    len(failed),
                )

        # 1. Hide the checkpoints in the recoverable staging area.
        txn_id, entries = self.checkpoints.stage_checkpoint_deletes(path)

        # 2. Remove the content itself.
        content_gone = False
        content_error = None
        try:
            self.delete_file(path)
        except Exception as exc:
            content_error = exc
        content_gone = not self.exists(path)

        if not content_gone:
            # Content survived (the failure happened before removal): undo
            # the checkpoint staging so both facts are back to normal.
            failed = self.checkpoints.rollback_checkpoint_deletes(txn_id, entries)
            if failed:
                # Compensation failed too: keep the transaction record so a
                # retry can finish the recovery.
                self.log.critical(
                    "Delete of %r failed and %d staged checkpoint(s) could not be "
                    "restored; retry the delete to recover",
                    path,
                    len(failed),
                    exc_info=content_error,
                )
            if content_error is not None:
                raise content_error
            raise OSError("content %r still exists after delete" % path)

        if content_error is not None:
            # The content really disappeared, but delete_file then raised
            # (the exact "file gone, storage then errored" case).  Do NOT
            # restore the checkpoints -- that would expose the old
            # checkpoint while the content is gone.  Keep the transaction
            # staged and surface the error; a retry resumes the cleanup and
            # reports a single, truthful completion.
            self.log.error(
                "Content %r was removed but delete reported an error %r; "
                "checkpoints stay staged until a retry finishes the delete",
                path,
                content_error,
            )
            raise content_error

        # 3. Permanently drop the staged checkpoints now that the content is
        #    really gone.  A failure here is retryable via the transaction
        #    record, so it must not look like a completed delete.
        try:
            self.checkpoints.commit_checkpoint_deletes(txn_id, entries)
            # Earlier attempts whose rollback could not fully restore their
            # checkpoints are now safe to purge for good.
            for adopted_txn_id in adopted_pending:
                if adopted_txn_id == txn_id:
                    continue
                self.checkpoints.resume_checkpoint_deletes(path, adopted_txn_id)
        except Exception as e:
            self.log.error(
                "Content %r was deleted but checkpoint cleanup failed; "
                "a retry will finish the operation",
                path,
                exc_info=True,
            )
            raise HTTPError(
                500,
                "Content %r was deleted but checkpoint cleanup failed: %s. "
                "Retry the request to complete the delete." % (path, e),
            ) from e

        self.emit(data={"action": "delete", "path": path})

    def rename(self, old_path, new_path):
        """项目内部接口说明。"""
        old_path = old_path.strip("/")
        new_path = new_path.strip("/")
        if old_path == new_path:
            return
        # Move checkpoints first; if the content move fails, move them back
        # so the old path keeps both its content and its checkpoints.
        moved_checkpoints = []
        for cp in self.checkpoints.list_checkpoints(old_path):
            self.checkpoints.rename_checkpoint(cp["id"], old_path, new_path)
            moved_checkpoints.append(cp)
        try:
            self.rename_file(old_path, new_path)
        except Exception:
            for cp in reversed(moved_checkpoints):
                try:
                    self.checkpoints.rename_checkpoint(cp["id"], new_path, old_path)
                except Exception:
                    self.log.error(
                        "Failed to move checkpoint %s back to %r",
                        cp["id"],
                        old_path,
                        exc_info=True,
                    )
            raise
        self.emit(data={"action": "rename", "path": new_path, "source_path": old_path})

    def update(self, model, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        new_path = model.get("path", path).strip("/")
        if path != new_path:
            self.rename(path, new_path)
        model = self.get(new_path, content=False)
        return model

    def info_string(self):
        """项目内部接口说明。"""
        return "Serving contents"

    def get_kernel_path(self, path, model=None):
        """项目内部接口说明。"""
        return ""

    def increment_filename(self, filename, path="", insert=""):
        """项目内部接口说明。"""
        # Extract the full suffix from the filename (e.g. .tar.gz)
        path = path.strip("/")
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        for i in itertools.count():
            insert_i = f"{insert}{i}" if i else ""
            name = f"{basename}{insert_i}{suffix}"
            if not self.exists(f"{path}/{name}"):
                break
        return name

    def validate_notebook_model(self, model, validation_error=None):
        """项目内部接口说明。"""
        try:
            # If we're given a validation_error dictionary, extract the exception
            # from it and raise the exception, else call nbformat's validate method
            # to determine if the notebook is valid.  This 'else' condition may
            # pertain to server extension not using the server's notebook read/write
            # functions.
            if validation_error is not None:
                e = validation_error.get("ValidationError")
                if isinstance(e, ValidationError):
                    raise e
            else:
                validate_nb(model["content"])
        except ValidationError as e:
            model["message"] = "Notebook validation failed: {}:\n{}".format(
                str(e),
                json.dumps(e.instance, indent=1, default=lambda obj: "<UNKNOWN>"),
            )
        return model

    def new_untitled(self, path="", type="", ext=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not self.dir_exists(path):
            raise HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = self.increment_filename(untitled + ext, path, insert=insert)
        path = f"{path}/{name}"
        return self.new(model, path)

    def new(self, model=None, path=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if model is None:
            model = {}

        if path.endswith(".ipynb"):
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        # no content, not a directory, so fill out new-file model
        if "content" not in model and model["type"] != "directory":
            if model["type"] == "notebook":
                model["content"] = new_notebook()
                model["format"] = "json"
            else:
                model["content"] = ""
                model["type"] = "file"
                model["format"] = "text"

        model = self.save(model, path)
        return model

    def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        path = from_path.strip("/")

        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = self.get(path)
        model.pop("path", None)
        model.pop("name", None)
        if model["type"] == "directory":
            raise HTTPError(400, "Can't copy directories")

        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if self.dir_exists(to_path):
            name = copy_pat.sub(".", from_name)
            to_name = self.increment_filename(name, to_path, insert="-Copy")
            to_path = f"{to_path}/{to_name}"
        elif is_destination_specified:
            if "/" in to_path:
                to_dir, to_name = to_path.rsplit("/", 1)
                if not self.dir_exists(to_dir):
                    raise HTTPError(404, "No such parent directory: %s to copy file in" % to_dir)
        else:
            raise HTTPError(404, "No such directory: %s" % to_path)

        model = self.save(model, to_path)
        self.emit(data={"action": "copy", "path": to_path, "source_path": from_path})
        return model

    def log_info(self):
        """项目内部接口说明。"""
        self.log.info(self.info_string())

    def trust_notebook(self, path):
        """项目内部接口说明。"""
        model = self.get(path)
        nb = model["content"]
        self.log.warning("Trusting notebook %s", path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    def check_and_sign(self, nb, path="", *, _retrying=False):
        """项目内部接口说明。"""
        try:
            if self.notary.check_cells(nb):
                self.notary.sign(nb)
            else:
                self.log.warning("Notebook %s is not trusted", path)
        except Exception:
            if _retrying:
                raise
            self.log.warning(
                "Signature store for notebook %s is corrupted or unavailable; "
                "recreating the store.",
                path,
                exc_info=True,
            )
            # The default implementation uses SQLiteSignatureStore if SQLite3 is available
            # and falls back to MemorySignatureStore if not; SQLiteSignatureStore will
            # attempt to recreate the database if it detects errors during initialization,
            # and fallback to in-memory (`:memory:`) SQLite database if necessary.
            self.notary.store = self.notary.store_factory()
            self.check_and_sign(nb, path, _retrying=True)

    def mark_trusted_cells(self, nb, path=""):
        """项目内部接口说明。"""
        trusted = self.notary.check_signature(nb)
        if not trusted:
            self.log.warning("Notebook %s is not trusted", path)
        self.notary.mark_cells(nb, trusted)

    def should_list(self, name):
        """项目内部接口说明。"""
        return not any(fnmatch(name, glob) for glob in self.hide_globs)

    # Part 3: Checkpoints API
    def create_checkpoint(self, path):
        """项目内部接口说明。"""
        return self.checkpoints.create_checkpoint(self, path)

    def restore_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        self.checkpoints.restore_checkpoint(self, checkpoint_id, path)

    def list_checkpoints(self, path):
        return self.checkpoints.list_checkpoints(path)

    def delete_checkpoint(self, checkpoint_id, path):
        return self.checkpoints.delete_checkpoint(checkpoint_id, path)


class AsyncContentsManager(ContentsManager):
    """项目内部接口说明。"""

    checkpoints_class = Type(AsyncCheckpoints, config=True)
    checkpoints = Instance(AsyncCheckpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default("checkpoints")
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default("checkpoints_kwargs")
    def _default_checkpoints_kwargs(self):
        return {
            "parent": self,
            "log": self.log,
        }

    # ContentsManager API part 1: methods that must be
    # implemented in subclasses.

    async def dir_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def is_hidden(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def file_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def exists(self, path):
        """项目内部接口说明。"""
        return await ensure_async(self.file_exists(path)) or await ensure_async(
            self.dir_exists(path)
        )

    async def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def save(self, model, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def delete_file(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    async def resolve_path(self, path: str) -> str | None:
        """项目内部接口说明。"""
        return None

    # ------------------------------------------------------------------
    # Delete serialization / recovery state (async counterpart)
    # ------------------------------------------------------------------

    def _async_delete_state(self):
        state = getattr(self, "_async_delete_bookkeeping", None)
        if state is None:
            state = {"locks": {}, "in_flight": set()}
            self._async_delete_bookkeeping = state
        return state

    async def _path_async_delete_lock(self, path):
        state = self._async_delete_state()
        lock = state["locks"].get(path)
        if lock is None:
            lock = asyncio.Lock()
            state["locks"][path] = lock
        return lock

    def _async_delete_was_active(self, path):
        state = self._async_delete_state()
        return path in state["in_flight"]

    def _mark_async_delete_active(self, path, active):
        state = self._async_delete_state()
        if active:
            state["in_flight"].add(path)
        else:
            state["in_flight"].discard(path)

    async def delete(self, path):
        """Async counterpart of :meth:`ContentsManager.delete`.

        Same recoverable ordering -- stage checkpoints, remove content,
        purge checkpoints -- and the same idempotent behavior for duplicate
        requests that race an in-flight delete.
        """
        path = path.strip("/")
        if not path:
            raise HTTPError(400, "Can't delete root")

        from anyio.to_thread import run_sync as _run_sync

        lock = await self._path_async_delete_lock(path)
        duplicate = self._async_delete_was_active(path)
        async with lock:
            self._mark_async_delete_active(path, True)
            try:
                claim, contended = await _run_sync(self._acquire_delete_claim, path)
                try:
                    await self._adelete_unlocked(
                        path, duplicate=duplicate or contended
                    )
                finally:
                    await _run_sync(self._release_delete_claim, claim)
            finally:
                self._mark_async_delete_active(path, False)

    async def _resume_interrupted_delete_async(self, path, pending):
        did_work = False
        ok = True
        for txn_id, _entries in pending:
            did_work = True
            if not await self.checkpoints.resume_checkpoint_deletes(path, txn_id):
                ok = False
        for _token, checkpoint_path in await self.checkpoints.list_checkpoint_files(path):
            if not os.path.lexists(checkpoint_path):
                continue
            did_work = True
            try:
                await self.checkpoints.purge_checkpoint_file(checkpoint_path)
            except FileNotFoundError:
                pass
            except Exception:
                self.log.error(
                    "Failed to purge stale checkpoint %s while resuming delete of %r",
                    checkpoint_path,
                    path,
                    exc_info=True,
                )
                ok = False
        if not did_work:
            return None
        return ok

    async def _adelete_unlocked(self, path, duplicate=False):
        content_exists = await ensure_async(self.exists(path))
        if not content_exists and duplicate:
            return

        if not getattr(self.checkpoints, "supports_delete_transactions", False):
            if not content_exists:
                raise HTTPError(404, "file or directory does not exist: %r" % path)
            await self.delete_file(path)
            await self.checkpoints.delete_all_checkpoints(path)
            self.emit(data={"action": "delete", "path": path})
            return

        pending = await self.checkpoints.find_delete_transactions(path)

        if not content_exists:
            resumed = await self._resume_interrupted_delete_async(path, pending)
            if resumed is None:
                if duplicate:
                    return
                raise HTTPError(404, "file or directory does not exist: %r" % path)
            if not resumed:
                raise HTTPError(
                    500,
                    "Delete of %r was interrupted and could not be fully completed; "
                    "the content is gone but some checkpoints remain. Retry the request."
                    % path,
                )
            self.emit(data={"action": "delete", "path": path})
            return

        adopted_pending = []
        for txn_id, entries in pending:
            failed = await self.checkpoints.rollback_checkpoint_deletes(txn_id, entries)
            adopted_pending.append(txn_id)
            if failed:
                self.log.error(
                    "Interrupted delete transaction %s for %r left %d checkpoint(s) "
                    "in the staging area; proceeding with a new delete",
                    txn_id,
                    path,
                    len(failed),
                )

        txn_id, entries = await self.checkpoints.stage_checkpoint_deletes(path)

        # 2. Remove the content itself.
        content_error = None
        try:
            await self.delete_file(path)
        except Exception as exc:
            content_error = exc
        content_gone = not await ensure_async(self.exists(path))

        if not content_gone:
            # Content survived: undo checkpoint staging so both facts match.
            failed = await self.checkpoints.rollback_checkpoint_deletes(txn_id, entries)
            if failed:
                self.log.critical(
                    "Delete of %r failed and %d staged checkpoint(s) could not be "
                    "restored; retry the delete to recover",
                    path,
                    len(failed),
                    exc_info=content_error,
                )
            if content_error is not None:
                raise content_error
            raise OSError("content %r still exists after delete" % path)

        if content_error is not None:
            # Content vanished but delete_file raised: keep checkpoints
            # staged and surface the error; a retry resumes the cleanup.
            self.log.error(
                "Content %r was removed but delete reported an error %r; "
                "checkpoints stay staged until a retry finishes the delete",
                path,
                content_error,
            )
            raise content_error

        try:
            await self.checkpoints.commit_checkpoint_deletes(txn_id, entries)
            for adopted_txn_id in adopted_pending:
                if adopted_txn_id == txn_id:
                    continue
                await self.checkpoints.resume_checkpoint_deletes(path, adopted_txn_id)
        except Exception as e:
            self.log.error(
                "Content %r was deleted but checkpoint cleanup failed; "
                "a retry will finish the operation",
                path,
                exc_info=True,
            )
            raise HTTPError(
                500,
                "Content %r was deleted but checkpoint cleanup failed: %s. "
                "Retry the request to complete the delete." % (path, e),
            ) from e

        self.emit(data={"action": "delete", "path": path})

    async def rename(self, old_path, new_path):
        """项目内部接口说明。"""
        old_path = old_path.strip("/")
        new_path = new_path.strip("/")
        if old_path == new_path:
            return
        moved_checkpoints = []
        for cp in await self.checkpoints.list_checkpoints(old_path):
            await self.checkpoints.rename_checkpoint(cp["id"], old_path, new_path)
            moved_checkpoints.append(cp)
        try:
            await self.rename_file(old_path, new_path)
        except Exception:
            for cp in reversed(moved_checkpoints):
                try:
                    await self.checkpoints.rename_checkpoint(cp["id"], new_path, old_path)
                except Exception:
                    self.log.error(
                        "Failed to move checkpoint %s back to %r",
                        cp["id"],
                        old_path,
                        exc_info=True,
                    )
            raise
        self.emit(data={"action": "rename", "path": new_path, "source_path": old_path})

    async def update(self, model, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        new_path = model.get("path", path).strip("/")
        if path != new_path:
            await self.rename(path, new_path)
        model = await self.get(new_path, content=False)
        return model

    async def increment_filename(self, filename, path="", insert=""):
        """项目内部接口说明。"""
        # Extract the full suffix from the filename (e.g. .tar.gz)
        path = path.strip("/")
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        for i in itertools.count():
            insert_i = f"{insert}{i}" if i else ""
            name = f"{basename}{insert_i}{suffix}"
            file_exists = await ensure_async(self.exists(f"{path}/{name}"))
            if not file_exists:
                break
        return name

    async def new_untitled(self, path="", type="", ext=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        dir_exists = await ensure_async(self.dir_exists(path))
        if not dir_exists:
            raise HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = await self.increment_filename(untitled + ext, path, insert=insert)
        path = f"{path}/{name}"
        return await self.new(model, path)

    async def new(self, model=None, path=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if model is None:
            model = {}

        if path.endswith(".ipynb"):
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        # no content, not a directory, so fill out new-file model
        if "content" not in model and model["type"] != "directory":
            if model["type"] == "notebook":
                model["content"] = new_notebook()
                model["format"] = "json"
            else:
                model["content"] = ""
                model["type"] = "file"
                model["format"] = "text"

        model = await self.save(model, path)
        return model

    async def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        path = from_path.strip("/")

        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = await self.get(path)
        model.pop("path", None)
        model.pop("name", None)
        if model["type"] == "directory":
            raise HTTPError(400, "Can't copy directories")

        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if await ensure_async(self.dir_exists(to_path)):
            name = copy_pat.sub(".", from_name)
            to_name = await self.increment_filename(name, to_path, insert="-Copy")
            to_path = f"{to_path}/{to_name}"
        elif is_destination_specified:
            if "/" in to_path:
                to_dir, to_name = to_path.rsplit("/", 1)
                if not await ensure_async(self.dir_exists(to_dir)):
                    raise HTTPError(404, "No such parent directory: %s to copy file in" % to_dir)
        else:
            raise HTTPError(404, "No such directory: %s" % to_path)

        model = await self.save(model, to_path)
        self.emit(data={"action": "copy", "path": to_path, "source_path": from_path})
        return model

    async def trust_notebook(self, path):
        """项目内部接口说明。"""
        model = await self.get(path)
        nb = model["content"]
        self.log.warning("Trusting notebook %s", path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    # Part 3: Checkpoints API
    async def create_checkpoint(self, path):
        """项目内部接口说明。"""
        return await self.checkpoints.create_checkpoint(self, path)

    async def restore_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        await self.checkpoints.restore_checkpoint(self, checkpoint_id, path)

    async def list_checkpoints(self, path):
        """项目内部接口说明。"""
        return await self.checkpoints.list_checkpoints(path)

    async def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        return await self.checkpoints.delete_checkpoint(checkpoint_id, path)
