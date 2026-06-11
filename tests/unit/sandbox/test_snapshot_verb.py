"""Unit coverage for ``DockerBackend.snapshot`` (durable session sandboxes, §5.2).

Drives the snapshot verb against a fake ``docker`` CLI dispatcher so the
lineage gate, the skip-empty floor (with the production containerd
``SizeRw == 4096`` no-write value), the flatten triggers (budget + depth),
and the env-keys scrub are all exercised without a real daemon.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aios.sandbox.backends.docker import _FLATTEN_DEPTH_CEILING, DockerBackend


class _FakeDocker:
    """A canned ``docker`` CLI + pipeline keyed on the subcommand.

    Models one container-under-snapshot (``parent_image`` / ``size_rw`` /
    ``container_labels``) and an image table (ref → id/size/depth/labels).
    ``commit`` and the flatten ``import`` both materialize the tag.
    """

    def __init__(self) -> None:
        self.parent_image = "img_S1"
        self.size_rw = 1_000_000
        self.container_labels: dict[str, str] = {}
        self.images: dict[str, dict[str, Any]] = {}
        self.calls: list[list[str]] = []
        self.pipelines: list[tuple[list[str], list[str]]] = []

    async def cli(self, argv: list[str], *, timeout_s: float = 30.0) -> tuple[int, bytes, bytes]:
        self.calls.append(argv)
        sub = argv[1]
        if sub == "stop":
            return 0, b"cid\n", b""
        if sub == "inspect" and "--size" in argv:
            out = f"{self.parent_image}\t{self.size_rw}\t{json.dumps(self.container_labels)}"
            return 0, out.encode(), b""
        if sub == "image" and len(argv) > 2 and argv[2] == "inspect":
            ref = argv[-1]
            img = self.images.get(ref)
            if img is None:
                return 1, b"", f"Error: No such image: {ref}".encode()
            # Real code inspects ``{{json .Config}}`` (the whole config), then
            # extracts ``.Labels`` — so emit a Config object, not bare labels.
            config = json.dumps({"Labels": img.get("labels", {})})
            out = f"{img['id']}\t{img['size']}\t{img['depth']}\t{config}"
            return 0, out.encode(), b""
        if sub == "commit":
            tag = argv[-1]
            self.images[tag] = {
                "id": "committed",
                "size": self.size_rw + 100_000,
                "depth": 2,
                "labels": dict(self.container_labels),
            }
            return 0, b"sha256:committed\n", b""
        raise AssertionError(f"unexpected docker cli: {argv}")

    async def pipeline(
        self, producer: list[str], consumer: list[str], *, timeout_s: float
    ) -> tuple[int, bytes, bytes]:
        self.pipelines.append((producer, consumer))
        tag = consumer[-1]
        self.images[tag] = {
            "id": "flattened",
            "size": 466_000_000,
            "depth": 1,
            "labels": {"aios.flattened": "true"},
        }
        return 0, b"sha256:flattened\n", b""


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> _FakeDocker:
    fd = _FakeDocker()
    monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", fd.cli)
    monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_pipeline", fd.pipeline)
    return fd


def _committed(fd: _FakeDocker) -> bool:
    return any(c[1] == "commit" for c in fd.calls)


# ── skip-empty floor (the containerd-store amendment) ────────────────────────


class TestSkipEmptyFloor:
    @pytest.mark.asyncio
    async def test_containerd_no_write_sizerw_4096_is_skipped(
        self, fake_docker: _FakeDocker
    ) -> None:
        """The production containerd image store reports ``SizeRw == 4096`` for
        a no-write container, NOT 0. With the default 8 KiB floor this must
        short-circuit to ``skipped_empty`` — otherwise read/chat-only sessions
        grow a chain every idle (§5.7)."""
        fake_docker.size_rw = 4096  # containerd no-write value
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "skipped_empty"
        assert out.image_id is None
        assert not _committed(fake_docker), "a no-write layer must not grow a chain"

    @pytest.mark.asyncio
    async def test_exactly_at_floor_is_skipped(self, fake_docker: _FakeDocker) -> None:
        fake_docker.size_rw = 8192
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "skipped_empty"
        assert not _committed(fake_docker)

    @pytest.mark.asyncio
    async def test_above_floor_commits(self, fake_docker: _FakeDocker) -> None:
        fake_docker.size_rw = 8193
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "committed"
        assert _committed(fake_docker)

    @pytest.mark.asyncio
    async def test_skip_empty_preserves_existing_tag(self, fake_docker: _FakeDocker) -> None:
        """A no-write release on a session that already has a snapshot keeps the
        prior tag canonical (image_id set), so the pointer heals to it."""
        fake_docker.size_rw = 4096
        fake_docker.images["tag:latest"] = {
            "id": "img_S1",  # == parent → lineage gate passes before the skip
            "size": 500_000,
            "depth": 2,
            "labels": {},
        }
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "skipped_empty"
        assert out.image_id == "img_S1"
        assert not _committed(fake_docker)


# ── lineage gate truth table ─────────────────────────────────────────────────


class TestLineageGate:
    @pytest.mark.asyncio
    async def test_tag_absent_commits_first_snapshot(self, fake_docker: _FakeDocker) -> None:
        # No tag in the image table → first snapshot → commit.
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "committed"

    @pytest.mark.asyncio
    async def test_tag_id_equals_corpse_parent_commits(self, fake_docker: _FakeDocker) -> None:
        """The corpse is a direct child of the current tag → committing advances
        the chain (the normal second-and-later idle cycle)."""
        fake_docker.parent_image = "img_S1"
        fake_docker.images["tag:latest"] = {
            "id": "img_S1",
            "size": 500_000,
            "depth": 2,
            "labels": {},
        }
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "committed"
        assert _committed(fake_docker)

    @pytest.mark.asyncio
    async def test_tag_moved_past_corpse_is_skipped_stale(self, fake_docker: _FakeDocker) -> None:
        """The tag is a child of the corpse's parent (crash-between-commit-and-rm,
        content-equal residue) → discard the corpse without a commit."""
        fake_docker.parent_image = "img_S1"
        fake_docker.images["tag:latest"] = {
            "id": "img_S2",  # tag has moved past the corpse's parent
            "size": 700_000,
            "depth": 3,
            "labels": {},
        }
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "skipped_stale"
        assert out.image_id == "img_S2"
        assert not _committed(fake_docker), "skipped_stale must never commit live-discarding work"


# ── flatten triggers ─────────────────────────────────────────────────────────


class TestFlattenTriggers:
    @pytest.mark.asyncio
    async def test_over_budget_flattens_not_commits(self, fake_docker: _FakeDocker) -> None:
        """Projected unique bytes over the per-session budget → flatten
        (the primary trigger on the containerd store, §1/§5.6)."""
        fake_docker.size_rw = 5_000_000_000  # 5 GB written
        out = await DockerBackend().snapshot(
            "cid",
            "tag:latest",
            empty_floor_bytes=8192,
            flatten_if_unique_bytes_over=4 * 1024 * 1024 * 1024,
        )
        assert out.kind == "flattened"
        assert fake_docker.pipelines, "flatten must run export|import"
        assert not _committed(fake_docker)

    @pytest.mark.asyncio
    async def test_depth_ceiling_flattens(self, fake_docker: _FakeDocker) -> None:
        """The soft depth backstop fires at the ceiling even under budget."""
        fake_docker.images["img_S1"] = {
            "id": "img_S1",
            "size": 500_000,
            "depth": _FLATTEN_DEPTH_CEILING - 1,  # +1 hits the ceiling
            "labels": {},
        }
        out = await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "flattened"
        assert fake_docker.pipelines

    @pytest.mark.asyncio
    async def test_under_budget_commits(self, fake_docker: _FakeDocker) -> None:
        fake_docker.size_rw = 1_000_000
        out = await DockerBackend().snapshot(
            "cid",
            "tag:latest",
            empty_floor_bytes=8192,
            flatten_if_unique_bytes_over=4 * 1024 * 1024 * 1024,
        )
        assert out.kind == "committed"
        assert not fake_docker.pipelines


# ── env-keys scrub scope (the verified container-bricker guard) ──────────────


class TestEnvKeysScrub:
    @pytest.mark.asyncio
    async def test_commit_scrubs_exactly_the_env_keys_label(self, fake_docker: _FakeDocker) -> None:
        """Commit emits ``--change 'ENV K='`` for exactly the names in
        ``aios.env_keys`` (read from the container labels), never the whole
        ``.Config.Env`` — scrubbing the base PATH/HOME bricks resume."""
        fake_docker.container_labels = {"aios.env_keys": "OPENAI_API_KEY,TOOL_BROKER_SECRET"}
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        commit = next(c for c in fake_docker.calls if c[1] == "commit")
        assert "ENV OPENAI_API_KEY=" in commit
        assert "ENV TOOL_BROKER_SECRET=" in commit
        # Only the labelled keys are scrubbed — no blanket PATH/HOME wipe.
        assert "ENV PATH=" not in commit
        assert "ENV HOME=" not in commit

    @pytest.mark.asyncio
    async def test_no_env_keys_label_means_plain_commit(self, fake_docker: _FakeDocker) -> None:
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        commit = next(c for c in fake_docker.calls if c[1] == "commit")
        assert "--change" not in commit


# ── flatten config restore ───────────────────────────────────────────────────


class TestFlattenConfigRestore:
    @pytest.mark.asyncio
    async def test_flatten_restores_workdir_home_cmd_not_path(
        self, fake_docker: _FakeDocker
    ) -> None:
        """Import strips all config; flatten restores WORKDIR/HOME/CMD and
        re-stamps labels, but deliberately NOT PATH (Docker injects a default,
        which keeps the restored CMD execable). §5.2."""
        fake_docker.container_labels = {
            "aios.base_image": "ghcr.io/eumemic/aios-sandbox:latest",
            "aios.session_id": "sess_x",
        }
        fake_docker.size_rw = 5_000_000_000
        await DockerBackend().snapshot(
            "cid",
            "tag:latest",
            empty_floor_bytes=8192,
            flatten_if_unique_bytes_over=1,  # force flatten
        )
        _producer, consumer = fake_docker.pipelines[0]
        joined = " ".join(consumer)
        assert "WORKDIR /workspace" in joined
        assert "ENV HOME=/home/aios" in joined
        assert 'CMD ["/usr/bin/tail","-f","/dev/null"]' in joined
        assert "aios.flattened=true" in joined
        assert "aios.base_image=ghcr.io/eumemic/aios-sandbox:latest" in joined
        # PATH is NOT restored.
        assert "ENV PATH=" not in joined
