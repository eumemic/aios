"""The snapshot-transport seam for durable session sandboxes (§5.2).

Horizontal scale — a second worker, host replacement/rebalancing — is a
stated requirement, so snapshot *transport* sits behind a small Protocol:
the lifecycle/salvage/lineage/flatten/GC-classify logic in
:mod:`aios.sandbox.registry` never sees how a snapshot is moved between
hosts. v1 ships :class:`LocalDaemonStore` only — an identity wrapper over
the local Docker daemon (today's behaviour exactly, now behind the seam),
so multi-host is *configuration plus a second store implementation*, not a
redesign.

A ``ref`` is a **host-independent name** — a pure function of (deployment,
session), never of which worker committed it. Each store maps the name
into its own namespace deterministically: ``LocalDaemonStore``'s name *is*
the local tag; a future ``RegistryStore`` would prefix its configured
registry path. The DB snapshot pointer therefore never needs rewriting
after a push.

**Verified-negative semantics** are load-bearing on :meth:`exists`/
:meth:`get`: only a *confirmed* not-found selects the resume cold-start
path. An indeterminate probe (daemon hiccup, timeout) raises rather than
reading as absence — treating a hiccup as "gone" would silently cold-start
a session and then let the next idle's lineage gate discard all post-hiccup
work as ``skipped_stale``.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from aios.sandbox.backends.base import SandboxBackend, SandboxBackendError


@runtime_checkable
class SnapshotStore(Protocol):
    """Pluggable transport for a session's canonical snapshot artifact."""

    def make_ref(self, session_id: str, local_image_tag: str) -> str:
        """Return an immutable ref for a newly committed generation."""
        ...

    async def put(self, local_image_tag: str, ref: str) -> str:
        """Persist a just-committed local image under ``ref``; return the stored ref.

        Identity for a local-only store (the image is already addressable);
        a remote store pushes/uploads here and returns the durable ref.
        """
        ...

    async def get(self, ref: str) -> str:
        """Make ``ref`` locally runnable and return the local tag to run.

        Pulls/loads for a remote store; identity for a local one. Raises on a
        ref that has vanished (callers gate with :meth:`exists` first).
        """
        ...

    async def exists(self, ref: str) -> bool:
        """Verified-negative existence: ``False`` only on a *confirmed*
        not-found; an indeterminate probe raises."""
        ...

    async def remove(self, ref: str) -> bool:
        """Remove the artifact. ``True`` on removed/already-gone, ``False`` on refusal."""
        ...

    async def size(self, ref: str) -> int:
        """Stored size in bytes (reporting only)."""
        ...


class LocalDaemonStore:
    """v1 identity store over the local Docker daemon.

    The committed image is already local under ``ref`` (the deterministic
    tag), so :meth:`put`/:meth:`get` are identity. ``exists``/``remove``/
    ``size`` map onto the backend's image verbs. A future
    ``RegistryStore``/``ObjectStore`` implements the same five methods with
    push/pull and drops in with no lifecycle changes.
    """

    def __init__(self, backend: SandboxBackend) -> None:
        self._backend = backend

    def make_ref(self, session_id: str, local_image_tag: str) -> str:
        return local_image_tag

    async def put(self, local_image_tag: str, ref: str) -> str:
        # Image is already local under ``ref``; nothing to transport in v1.
        return ref

    async def get(self, ref: str) -> str:
        if not await self.exists(ref):
            # Should-never-happen: callers verify existence first. A vanish
            # between the check and here is a race, not a silent base fallback.
            raise SandboxBackendError(f"snapshot ref vanished during resolve: {ref}")
        return ref

    async def exists(self, ref: str) -> bool:
        # ``image_labels`` is verified-negative at the backend boundary
        # (``None`` ⇒ confirmed-absent, indeterminate ⇒ raises), which is
        # exactly the semantics this seam promises.
        return await self._backend.image_labels(ref) is not None

    async def remove(self, ref: str) -> bool:
        return await self._backend.remove_image(ref)

    async def size(self, ref: str) -> int:
        return await self._backend.image_size(ref)


class TarballStore:
    """Canonical immutable Docker archives on a persistent host filesystem."""

    _FORMAT_VERSION = 1

    def __init__(self, backend: SandboxBackend, root: Path) -> None:
        self._backend = backend
        self._root = root.resolve()
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)

    def preflight(self, *, headroom_bytes: int = 0) -> None:
        """Fail loudly unless root is mounted, writable and durably syncable."""
        if not os.path.ismount(self._root):
            raise SandboxBackendError(f"snapshot store root is not a mount: {self._root}")
        probe = self._root / f".preflight-{uuid.uuid4().hex}"
        try:
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, b"aios snapshot preflight\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            dfd = os.open(self._root, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
            stat = os.statvfs(self._root)
            free = stat.f_bavail * stat.f_frsize
            if free < headroom_bytes:
                raise SandboxBackendError(
                    f"snapshot store below required headroom: {free} < {headroom_bytes}"
                )
        finally:
            probe.unlink(missing_ok=True)

    def artifacts(self) -> list[tuple[str, float]]:
        """Return immutable published artifact refs and mtimes (never temp files)."""
        result: list[tuple[str, float]] = []
        for path in self._root.rglob("*.tar"):
            if path.name.startswith(".") or path.is_symlink():
                continue
            result.append((str(path.relative_to(self._root)), path.stat().st_mtime))
        return result

    def used_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in self._root.rglob("*.tar") if not path.is_symlink()
        )

    def make_ref(self, session_id: str, local_image_tag: str) -> str:
        del local_image_tag
        return f"{session_id}/{uuid.uuid4().hex}.tar"

    def _path(self, ref: str) -> Path:
        path = self._root.joinpath(ref)
        if (
            Path(ref).is_absolute()
            or ".." in Path(ref).parts
            or path.resolve().parent == self._root.parent
        ):
            raise SandboxBackendError("invalid snapshot ref")
        try:
            path.resolve().relative_to(self._root)
        except ValueError as err:
            raise SandboxBackendError("invalid snapshot ref") from err
        return path

    def _manifest_path(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".json")

    def _legacy(self, ref: str) -> bool:
        return "/" not in ref and not ref.endswith(".tar")

    async def put(self, local_image_tag: str, ref: str) -> str:
        path = self._path(ref)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        manifest_temp = temp.with_suffix(temp.suffix + ".json")
        try:
            await self._backend.save_image(local_image_tag, temp)
            digest, size = self._digest(temp)
            with temp.open("rb") as artifact:
                os.fsync(artifact.fileno())
            manifest = {
                "format_version": self._FORMAT_VERSION,
                "local_tag": local_image_tag,
                "stored_bytes": size,
                "sha256": digest,
            }
            with manifest_temp.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            # Verify the exact fsynced bytes before exposing either immutable name.
            verify_digest, verify_size = self._digest(temp)
            if verify_digest != digest or verify_size != size:
                raise SandboxBackendError("snapshot artifact integrity check failed before rename")
            os.replace(temp, path)
            os.replace(manifest_temp, self._manifest_path(path))
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._verify(path)
            return ref
        finally:
            temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)

    async def get(self, ref: str) -> str:
        if self._legacy(ref):
            return await LocalDaemonStore(self._backend).get(ref)
        path = self._path(ref)
        manifest = self._verify(path)
        local_tag = str(manifest["local_tag"])
        if await self._backend.image_labels(local_tag) is None:
            await self._backend.load_image(path)
        return local_tag

    async def exists(self, ref: str) -> bool:
        if self._legacy(ref):
            return await LocalDaemonStore(self._backend).exists(ref)
        path = self._path(ref)
        if not path.exists() and not self._manifest_path(path).exists():
            return False
        self._verify(path)
        return True

    async def remove(self, ref: str) -> bool:
        if self._legacy(ref):
            return await LocalDaemonStore(self._backend).remove(ref)
        path = self._path(ref)
        path.unlink(missing_ok=True)
        self._manifest_path(path).unlink(missing_ok=True)
        return True

    async def size(self, ref: str) -> int:
        if self._legacy(ref):
            return await LocalDaemonStore(self._backend).size(ref)
        return cast(int, self._verify(self._path(ref))["stored_bytes"])

    @staticmethod
    def _digest(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as err:
            raise SandboxBackendError(f"snapshot artifact I/O failed: {err}") from err
        return digest.hexdigest(), size

    def _verify(self, path: Path) -> dict[str, object]:
        try:
            manifest = cast(
                dict[str, object],
                json.loads(self._manifest_path(path).read_text(encoding="utf-8")),
            )
            digest, size = self._digest(path)
            if (
                manifest.get("format_version") != self._FORMAT_VERSION
                or manifest.get("sha256") != digest
                or manifest.get("stored_bytes") != size
                or not isinstance(manifest.get("local_tag"), str)
            ):
                raise SandboxBackendError("snapshot artifact integrity check failed")
            return manifest
        except SandboxBackendError:
            raise
        except (OSError, ValueError, TypeError) as err:
            raise SandboxBackendError(f"snapshot artifact integrity check failed: {err}") from err
