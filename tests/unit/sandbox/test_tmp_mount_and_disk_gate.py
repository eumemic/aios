"""The ``/tmp`` bind mount and the snapshot headroom gate (#2280).

A durable session's snapshot is ``docker export``/``docker commit`` of its
rootfs, so anything written outside a bind mount is preserved forever. One
production session reached 18 GiB of which 16.4 GiB was ``/tmp`` scratch.

Two properties are asserted here:

* ``/tmp`` is a per-session bind mount for SESSIONS ONLY — runs and browser
  planes are bare-destroyed, so binding their ``/tmp`` would make scratch
  outlive the container instead of dying with it.
* The headroom gate covers BOTH snapshot verbs, and sizes a flatten on what
  the export will actually contain. Sizing it on the fat image would refuse
  the very flatten that makes the image thin — a fail-closed state whose only
  exit is the operation it just refused (``patterns/in-band-exit``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.backends.docker import DockerBackend

from .test_snapshot_verb import _FakeDocker, _set_free_disk
from .test_snapshot_verb import fake_docker as _fake_docker

# Re-export the shared fixture under its usual name for this module.
fake_docker = _fake_docker

GIB = 1024**3


# ── the mount ────────────────────────────────────────────────────────────────


def _assemble(session_id: str) -> object:
    from aios.sandbox.spec import _assemble_plan

    with (
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/a"),
        ),
        patch("aios.sandbox.volumes.ensure_session_uploads_dir", return_value=Path("/tmp/u")),
        patch(
            "aios.sandbox.volumes.ensure_session_tmp_dir",
            return_value=Path(f"/ws/_tmp/{session_id}"),
        ),
    ):
        return _assemble_plan(
            session_id=session_id,
            instance_id="default",
            image="img:latest",
            workspace_path=Path("/ws/sess"),
            snapshot_ref=None,
            snapshot_budget_bytes=None,
            env_config=None,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            tool_broker_url="http://aios-worker:54321",
            tool_broker_secret="s",
            tool_socket_host_path=None,
        )


def _tmp_mounts(plan: object) -> list[object]:
    return [m for m in plan.spec.extra_mounts if m.sandbox_path == "/tmp"]  # type: ignore[attr-defined]


class TestTmpMount:
    def test_session_gets_a_writable_tmp_bind(self) -> None:
        mounts = _tmp_mounts(_assemble("sess_01TEST"))
        assert len(mounts) == 1
        assert mounts[0].host_path == Path("/ws/_tmp/sess_01TEST")  # type: ignore[attr-defined]
        assert mounts[0].read_only is False  # type: ignore[attr-defined]

    @pytest.mark.parametrize("owner", ["wfr_01TEST", "acc_01TEST"])
    def test_bare_destroy_owners_keep_tmp_in_the_writable_layer(self, owner: str) -> None:
        """A run and a browser plane are destroyed whole, so their ``/tmp`` is
        already reclaimed. Binding it would make it outlive the container."""
        assert _tmp_mounts(_assemble(owner)) == []

    def test_unrecognised_owner_id_does_not_raise(self) -> None:
        """The prefix test must not be ``sandbox_owner_kind`` — that helper is
        exhaustive-and-raising, and an odd owner id must not become a failed
        provision."""
        assert _tmp_mounts(_assemble("run_01LEGACY")) == []

    def test_tmp_is_the_only_new_mount(self) -> None:
        """Guards the mount-drift key: every added mount recycles every live
        sandbox on the next touch, so an accidental extra is not free."""
        paths = {m.sandbox_path for m in _assemble("sess_01TEST").spec.extra_mounts}  # type: ignore[attr-defined]
        assert paths == {"/mnt/attachments", "/mnt/uploads", "/tmp"}


# ── the headroom gate ────────────────────────────────────────────────────────


class TestSnapshotDiskGate:
    @pytest.mark.asyncio
    async def test_commit_is_gated_too(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before #2280 only the flatten branch had a floor under it, so the
        path taken by SMALLER sandboxes could consume the last free byte."""
        fake_docker.size_rw = 2 * GIB
        _set_free_disk(monkeypatch, 1_000_000)  # 1 MB free
        with pytest.raises(SandboxBackendError, match="commit deferred"):
            await DockerBackend().snapshot(
                "cid",
                "tag:latest",
                empty_floor_bytes=8192,
                flatten_if_unique_bytes_over=100 * GIB,  # stay on the commit path
            )
        assert not any(c[1] == "commit" for c in fake_docker.calls), (
            "a deferred commit must never start the commit"
        )

    @pytest.mark.asyncio
    async def test_commit_proceeds_with_headroom(
        self, fake_docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_docker.size_rw = 2 * GIB
        _set_free_disk(monkeypatch, 100 * GIB)
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=100 * GIB
        )
        assert any(c[1] == "commit" for c in fake_docker.calls)

    @pytest.mark.asyncio
    async def test_flatten_estimate_excludes_what_the_filter_drops(
        self,
        fake_docker: _FakeDocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """THE in-band-exit case. A 25 GB session whose bulk is ``/tmp`` needs
        headroom for its FILTERED output (~small), not its fat rootfs. Sizing
        on the fat figure would refuse the one operation that shrinks it, and
        the session could never recover on a full disk."""
        fake_docker.size_rw = 25 * GIB
        fake_docker.ephemeral_bytes = 23 * GIB  # nearly all of it is scratch
        # Enough for the filtered ~2 GB (x1.75) + the 15 GiB floor, but far
        # short of the unfiltered 25 GB (x1.75) + floor.
        _set_free_disk(monkeypatch, 20 * GIB)

        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=4 * GIB
        )

        assert fake_docker.pipelines, "the flatten must proceed on the filtered estimate"

    @pytest.mark.asyncio
    async def test_flatten_still_refused_when_even_filtered_output_does_not_fit(
        self,
        fake_docker: _FakeDocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The gate is relaxed by the filter, not removed by it."""
        fake_docker.size_rw = 25 * GIB
        fake_docker.ephemeral_bytes = 1 * GIB
        _set_free_disk(monkeypatch, 2 * GIB)
        with pytest.raises(SandboxBackendError, match="flatten deferred"):
            await DockerBackend().snapshot(
                "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=4 * GIB
            )
        assert not fake_docker.pipelines

    @pytest.mark.asyncio
    async def test_ephemeral_probe_only_counts_the_container_rootfs(self) -> None:
        """After #2280 ``/tmp`` is a BIND MOUNT, and ``docker export`` omits
        bind-mount contents — so those bytes are not in the stream and the
        filter will not drop them. Counting them would make the gate subtract
        space the export never needed and start a flatten with too little
        disk. The probe must compare device numbers, not just run ``du``."""
        captured: list[list[str]] = []

        async def _fake_cli(argv: list[str], **_kw: object) -> tuple[int, bytes, bytes]:
            captured.append(argv)
            return 0, b"", b""  # every prefix is a mount ⇒ nothing to drop

        with patch("aios.sandbox.backends.docker.run_docker_cli", _fake_cli):
            assert await DockerBackend()._ephemeral_bytes("cid") == 0

        script = captured[0][-1]
        assert "stat -c %d /" in script, "must establish the rootfs device"
        assert "du -sbx" in script
        for prefix in ("tmp", "var/tmp", "run"):
            assert f"'/{prefix}'" in script

    @pytest.mark.asyncio
    async def test_ephemeral_probe_sums_every_reported_prefix(self) -> None:
        async def _fake_cli(_argv: list[str], **_kw: object) -> tuple[int, bytes, bytes]:
            return 0, b"1000\t/tmp\n250\t/var/tmp\n30\t/run\n", b""

        with patch("aios.sandbox.backends.docker.run_docker_cli", _fake_cli):
            assert await DockerBackend()._ephemeral_bytes("cid") == 1280

    @pytest.mark.asyncio
    async def test_ephemeral_probe_failure_is_swallowed(self, fake_docker: _FakeDocker) -> None:
        """``_ephemeral_bytes`` degrades to 0 rather than propagating: refusing
        to snapshot is how a session gets stranded."""
        backend = DockerBackend()
        with patch(
            "aios.sandbox.backends.docker.run_docker_cli",
            side_effect=SandboxBackendError("no exec"),
        ):
            assert await backend._ephemeral_bytes("cid") == 0

    @pytest.mark.asyncio
    async def test_gate_measures_the_configured_filesystem(
        self,
        fake_docker: _FakeDocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The gate must stat the docker graph root's filesystem, not whatever
        ``/`` happens to be inside the worker container."""
        seen: list[str] = []

        def _usage(path: str) -> MagicMock:
            seen.append(path)
            return MagicMock(free=100 * GIB)

        monkeypatch.setattr("aios.sandbox.backends.docker.shutil.disk_usage", _usage)
        monkeypatch.setattr(
            "aios.sandbox.backends.docker.get_settings",
            lambda: _settings_with(sandbox_disk_stat_path="/var/lib/docker"),
        )
        await DockerBackend().snapshot(
            "cid", "tag:latest", empty_floor_bytes=8192, flatten_if_unique_bytes_over=100 * GIB
        )
        assert seen == ["/var/lib/docker"]


def _settings_with(**overrides: object) -> object:
    from aios.config import get_settings

    real = get_settings()
    stub = MagicMock(wraps=real)
    for key in (
        "sandbox_flatten_disk_floor_bytes",
        "sandbox_inspect_size_timeout_seconds",
        "sandbox_pipeline_stall_seconds",
        "sandbox_snapshot_timeout_floor_seconds",
        "sandbox_snapshot_timeout_retry_multiplier",
        "sandbox_snapshot_timeout_retry_cap",
        "sandbox_snapshot_throughput_state_path",
        "sandbox_docker_cli_timeout_seconds",
        "sandbox_disk_stat_path",
    ):
        setattr(stub, key, getattr(real, key))
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub
