"""Unit coverage for the ``SnapshotStore`` seam (durable session sandboxes, §5.2).

``LocalDaemonStore`` is the v1 identity wrapper over the local daemon: put/get
are identity, exists/remove/size map onto the backend's image verbs. The
load-bearing property under test is **verified-negative** existence — a
confirmed not-found returns False, an indeterminate probe raises (never reads
as absence, which would silently cold-start a session and then let the next
idle's lineage gate discard its post-hiccup work).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.snapshot_store import LocalDaemonStore
from tests.helpers.sandbox import FakeBackend


class TestLocalDaemonStore:
    @pytest.mark.asyncio
    async def test_put_is_identity(self) -> None:
        store = LocalDaemonStore(FakeBackend())
        assert await store.put("local-tag:latest", "ref:latest") == "ref:latest"

    @pytest.mark.asyncio
    async def test_exists_true_when_present(self) -> None:
        backend = FakeBackend()
        backend.image_labels_by_ref["ref:latest"] = {"aios.managed": "true"}
        store = LocalDaemonStore(backend)
        assert await store.exists("ref:latest") is True

    @pytest.mark.asyncio
    async def test_exists_true_even_with_empty_labels(self) -> None:
        """An image with no labels still exists ({} is not None)."""
        backend = FakeBackend()
        backend.image_labels_by_ref["ref:latest"] = {}
        store = LocalDaemonStore(backend)
        assert await store.exists("ref:latest") is True

    @pytest.mark.asyncio
    async def test_exists_false_on_verified_not_found(self) -> None:
        backend = FakeBackend()  # ref absent ⇒ image_labels returns None
        store = LocalDaemonStore(backend)
        assert await store.exists("missing:latest") is False

    @pytest.mark.asyncio
    async def test_exists_raises_on_indeterminate_probe(self) -> None:
        """A daemon hiccup must propagate, never read as absence."""
        backend = FakeBackend()
        backend.image_labels = AsyncMock(  # type: ignore[method-assign]
            side_effect=SandboxBackendError("daemon hiccup")
        )
        store = LocalDaemonStore(backend)
        with pytest.raises(SandboxBackendError, match="daemon hiccup"):
            await store.exists("ref:latest")

    @pytest.mark.asyncio
    async def test_get_returns_ref_when_present(self) -> None:
        backend = FakeBackend()
        backend.image_labels_by_ref["ref:latest"] = {}
        store = LocalDaemonStore(backend)
        assert await store.get("ref:latest") == "ref:latest"

    @pytest.mark.asyncio
    async def test_get_raises_when_vanished(self) -> None:
        store = LocalDaemonStore(FakeBackend())
        with pytest.raises(SandboxBackendError, match="vanished"):
            await store.get("missing:latest")

    @pytest.mark.asyncio
    async def test_remove_delegates_to_backend(self) -> None:
        backend = FakeBackend()
        backend.image_labels_by_ref["ref:latest"] = {}
        store = LocalDaemonStore(backend)
        assert await store.remove("ref:latest") is True
        assert "ref:latest" in backend.removed_image_refs

    @pytest.mark.asyncio
    async def test_remove_reports_refusal(self) -> None:
        backend = FakeBackend()
        backend.refuse_remove_refs.add("ref:latest")
        store = LocalDaemonStore(backend)
        assert await store.remove("ref:latest") is False

    @pytest.mark.asyncio
    async def test_size_delegates_to_backend(self) -> None:
        backend = FakeBackend()
        backend.image_sizes_by_ref["ref:latest"] = 12345
        store = LocalDaemonStore(backend)
        assert await store.size("ref:latest") == 12345
