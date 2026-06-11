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

from typing import Protocol, runtime_checkable

from aios.sandbox.backends.base import SandboxBackend, SandboxBackendError


@runtime_checkable
class SnapshotStore(Protocol):
    """Pluggable transport for a session's canonical snapshot artifact."""

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
