"""Unit coverage for ``DockerBackend.snapshot`` (durable session sandboxes, §5.2).

Drives the snapshot verb against a fake ``docker`` CLI dispatcher so the
lineage gate, the skip-empty floor (with the production containerd
``SizeRw == 4096`` no-write value), the flatten triggers (budget + depth),
and the env-keys scrub are all exercised without a real daemon.
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from aios.config import get_settings
from aios.sandbox.backends.base import SandboxBackendError, SandboxSnapshotTimeoutError
from aios.sandbox.backends.docker import _FLATTEN_DEPTH_CEILING, DockerBackend

_Usage = namedtuple("_Usage", ["total", "used", "free"])


def _set_free_disk(monkeypatch: pytest.MonkeyPatch, free_bytes: int) -> None:
    """Pin ``shutil.disk_usage(...).free`` the flatten disk-gate reads, so the
    gate is deterministic instead of depending on the host/runner's real disk."""
    monkeypatch.setattr(
        "aios.sandbox.backends.docker.shutil.disk_usage",
        lambda _path: _Usage(total=free_bytes * 2, used=free_bytes, free=free_bytes),
    )


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
        self.cli_timeouts: list[tuple[list[str], float]] = []
        self.pipelines: list[tuple[list[str], list[str]]] = []
        # (stall_timeout_s, max_timeout_s) each flatten ran with — lets a test
        # assert the pipeline uses the config-driven progress deadlines, not a
        # size-scaled timeout.
        self.pipeline_timeouts: list[tuple[float, float]] = []

    async def cli(
        self, argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        self.calls.append(argv)
        self.cli_timeouts.append((argv, timeout_s))
        sub = argv[1]
        if sub == "stop":
            return 0, b"cid\n", b""
        if sub == "inspect" and "--size" in argv:
            # The budgeted writable-layer size-walk: `{{.SizeRw}}` only.
            return 0, f"{self.size_rw}".encode(), b""
        if sub == "inspect":
            # The cheap metadata probe: `{{.Image}}\t{{json .Config.Labels}}`.
            out = f"{self.parent_image}\t{json.dumps(self.container_labels)}"
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
        self,
        producer: list[str],
        consumer: list[str],
        *,
        stall_timeout_s: float,
        max_timeout_s: float,
    ) -> tuple[int, bytes, bytes]:
        self.pipelines.append((producer, consumer))
        self.pipeline_timeouts.append((stall_timeout_s, max_timeout_s))
        tag = consumer[-1]
        self.images[tag] = {
            "id": "flattened",
            "size": 466_000_000,
            "depth": 1,
            "labels": {"aios.flattened": "true"},
        }
        return 0, b"sha256:flattened\n", b""


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeDocker:
    fd = _FakeDocker()
    monkeypatch.setattr(
        get_settings(), "sandbox_snapshot_throughput_state_path", tmp_path / "throughput.json"
    )
    monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", fd.cli)
    monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_pipeline", fd.pipeline)
    # Default to abundant free disk so flatten-path tests are deterministic;
    # the disk-gate tests override this explicitly.
    _set_free_disk(monkeypatch, 1024**4)  # 1 TiB
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


class TestSnapshotTimeoutBudget:
    @pytest.mark.asyncio
    async def test_large_commit_uses_size_derived_budget(self, fake_docker: _FakeDocker) -> None:
        fake_docker.size_rw = 12_000_000_000
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        commit_timeout = next(
            timeout for argv, timeout in fake_docker.cli_timeouts if argv[1] == "commit"
        )
        assert commit_timeout == 480.0

    @pytest.mark.asyncio
    async def test_inspect_size_walk_gets_generous_timeout(self, fake_docker: _FakeDocker) -> None:
        """The ``docker inspect --size`` writable-layer walk must NOT run on the
        blanket 30s CLI bound. ``--size`` stat-walks the corpse's RW layer, so a
        large corpse would time out here before commit is even reached — the one
        size-scaling call #1838 left unbudgeted — wedging salvage (the breaker's
        half-open re-probe then re-hits the same 30s wall forever).
        """
        fake_docker.size_rw = 12_000_000_000  # large, slow-to-walk corpse
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        size_walk_timeout = next(
            t for argv, t in fake_docker.cli_timeouts if argv[1] == "inspect" and "--size" in argv
        )
        assert size_walk_timeout == get_settings().sandbox_inspect_size_timeout_seconds

    @pytest.mark.asyncio
    async def test_inspect_metadata_stays_on_blanket_timeout(
        self, fake_docker: _FakeDocker
    ) -> None:
        """The cheap metadata inspect (parent image + labels, no ``--size``) is a
        management call and stays on the blanket bound — only the data-walking
        ``--size`` probe is budgeted. This is the split that makes the module
        comment's 'metadata on the blanket bound, data-walk budgeted' literally true.
        """
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        meta_timeout = next(
            t
            for argv, t in fake_docker.cli_timeouts
            if argv[1] == "inspect" and "--size" not in argv
        )
        assert meta_timeout == 30.0


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
        re-stamps labels, but deliberately NOT PATH. The restored CMD invokes
        ``tail`` by absolute path (``/usr/bin/tail``), so it no longer depends
        on PATH resolution at all. §5.2."""
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
        assert "ENV HOME=/root" in joined
        assert 'CMD ["/usr/bin/tail","-f","/dev/null"]' in joined
        assert "aios.flattened=true" in joined
        assert "aios.base_image=ghcr.io/eumemic/aios-sandbox:latest" in joined
        # PATH is NOT restored.
        assert "ENV PATH=" not in joined


# ── flatten disk-admission gate + the >5 GB salvage regime ───────────────────


class TestFlattenDiskGate:
    """The flatten path first checks free disk against the estimated transient
    ``docker export`` tar cost plus a floor, and DEFERS (raises, corpse
    retained) rather than fail mid-write into a half-baked image. The deferral
    is what the salvage breaker counts. With sufficient disk the flatten runs
    on the progress-aware pipeline — the fix for the size regime the old
    size-scaled timeout could never satisfy."""

    @pytest.mark.asyncio
    async def test_flatten_deferred_when_disk_below_required(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_docker.size_rw = 6_000_000_000  # 6 GB writable layer → over budget
        _set_free_disk(monkeypatch, 1_000_000_000)  # only 1 GB free
        with pytest.raises(SandboxBackendError, match="flatten deferred"):
            await DockerBackend().snapshot(
                "cid",
                "tag:latest",
                empty_floor_bytes=8192,
                flatten_if_unique_bytes_over=4 * 1024 * 1024 * 1024,
            )
        assert not fake_docker.pipelines, "a deferred flatten must never start the pipeline"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size_rw", [6_000_000_000, 200_000_000_000])
    async def test_large_corpse_flatten_uses_size_derived_deadline(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch, size_rw: int
    ) -> None:
        """Flatten receives the same SizeRw-derived snapshot budget as commit."""
        fake_docker.size_rw = size_rw
        _set_free_disk(monkeypatch, 2 * 1024**5)  # 2 PiB — over required for any size here
        out = await DockerBackend().snapshot(
            "cid",
            "tag:latest",
            empty_floor_bytes=8192,
            flatten_if_unique_bytes_over=4 * 1024 * 1024 * 1024,
        )
        assert out.kind == "flattened"
        assert fake_docker.pipelines, "a large corpse must flatten via the pipeline"
        settings = get_settings()
        budget = max(60.0, size_rw * 20e-9) * 2.0
        assert fake_docker.pipeline_timeouts == [
            (min(settings.sandbox_pipeline_stall_seconds, budget), budget)
        ]


# ── timeout retry-state lifecycle (#2009 review) ─────────────────────────────


class TestTimeoutRetryStateCleared:
    """Every SUCCESSFUL snapshot outcome must clear the corpse's timeout-retry
    entry — not just the commit/flatten paths.

    The normal timeout recovery is ``docker commit`` completing server-side
    while the client's deadline fires: the retry then observes the advanced
    tag and returns ``skipped_stale`` (or ``skipped_empty``) from an early
    return, and the corpse is removed immediately after. If those paths didn't
    clear the entry, nothing ever could — the container is gone — so
    ``_snapshot_timeout_attempts`` would grow one stranded entry per
    successfully-salvaged timeout corpse for the lifetime of the worker.
    """

    @pytest.mark.asyncio
    async def test_timeout_then_skipped_stale_clears_retry_state(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = DockerBackend()
        fake_docker.size_rw = 12_000_000_000
        fake_docker.parent_image = "img_S1"

        real_cli = fake_docker.cli

        async def timing_out_commit(
            argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
        ) -> tuple[int, bytes, bytes]:
            if argv[1] == "commit":
                raise SandboxSnapshotTimeoutError(f"docker cli timed out after {timeout_s}s")
            return await real_cli(argv, timeout_s=timeout_s, snapshot_timeout=snapshot_timeout)

        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", timing_out_commit)

        with pytest.raises(SandboxSnapshotTimeoutError):
            await backend.snapshot(
                "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
            )
        assert backend._snapshot_timeout_attempts["cid"] == 1, "a timeout must escalate the budget"

        # The commit actually landed daemon-side: the tag has moved past the
        # corpse's parent, so the retry takes the lineage-gate early return.
        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", real_cli)
        fake_docker.images["tag:latest"] = {
            "id": "img_S2",
            "size": 700_000,
            "depth": 3,
            "labels": {},
        }

        out = await backend.snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )

        assert out.kind == "skipped_stale"
        assert "cid" not in backend._snapshot_timeout_attempts, (
            "skipped_stale is a successful salvage — it must not strand retry state"
        )
        assert backend._snapshot_timeout_attempts == {}

    @pytest.mark.asyncio
    async def test_timeout_then_skipped_empty_clears_retry_state(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = DockerBackend()
        fake_docker.size_rw = 12_000_000_000

        real_cli = fake_docker.cli

        async def timing_out_commit(
            argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
        ) -> tuple[int, bytes, bytes]:
            if argv[1] == "commit":
                raise SandboxSnapshotTimeoutError(f"docker cli timed out after {timeout_s}s")
            return await real_cli(argv, timeout_s=timeout_s, snapshot_timeout=snapshot_timeout)

        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", timing_out_commit)

        with pytest.raises(SandboxSnapshotTimeoutError):
            await backend.snapshot(
                "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
            )
        assert backend._snapshot_timeout_attempts["cid"] == 1

        # Retry against a corpse whose writable layer is at/below the floor →
        # the identity short-circuit early return.
        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", real_cli)
        fake_docker.size_rw = 4096

        out = await backend.snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )

        assert out.kind == "skipped_empty"
        assert not _committed(fake_docker)
        assert "cid" not in backend._snapshot_timeout_attempts, (
            "skipped_empty is a successful salvage — it must not strand retry state"
        )

    @pytest.mark.asyncio
    async def test_repeated_timeouts_escalate_then_success_clears(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Consecutive timeouts accumulate (escalating the budget); the first
        successful outcome settles the entry back to empty."""
        backend = DockerBackend()
        fake_docker.size_rw = 12_000_000_000
        real_cli = fake_docker.cli
        commit_timeouts: list[float] = []

        async def timing_out_commit(
            argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
        ) -> tuple[int, bytes, bytes]:
            if argv[1] == "commit":
                commit_timeouts.append(timeout_s)
                raise SandboxSnapshotTimeoutError(f"docker cli timed out after {timeout_s}s")
            return await real_cli(argv, timeout_s=timeout_s, snapshot_timeout=snapshot_timeout)

        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", timing_out_commit)
        for expected_attempts in (1, 2, 3):
            with pytest.raises(SandboxSnapshotTimeoutError):
                await backend.snapshot(
                    "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
                )
            assert backend._snapshot_timeout_attempts["cid"] == expected_attempts

        assert commit_timeouts == sorted(commit_timeouts), "each retry must get a larger budget"
        assert commit_timeouts[1] > commit_timeouts[0]

        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", real_cli)
        out = await backend.snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out.kind == "committed"
        assert backend._snapshot_timeout_attempts == {}

    @pytest.mark.asyncio
    async def test_non_timeout_failure_leaves_retry_state_untouched(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only typed snapshot timeouts escalate. A generic backend error is a
        retained-corpse failure that must neither escalate nor clear."""
        backend = DockerBackend()
        fake_docker.size_rw = 12_000_000_000
        real_cli = fake_docker.cli

        async def failing_commit(
            argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
        ) -> tuple[int, bytes, bytes]:
            if argv[1] == "commit":
                return 1, b"", b"no space left on device"
            return await real_cli(argv, timeout_s=timeout_s, snapshot_timeout=snapshot_timeout)

        monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", failing_commit)
        with pytest.raises(SandboxBackendError):
            await backend.snapshot(
                "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
            )
        assert "cid" not in backend._snapshot_timeout_attempts
