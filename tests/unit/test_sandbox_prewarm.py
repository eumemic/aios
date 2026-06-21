"""Unit tests for the operator prewarm optimization (#1348).

Two surfaces, no Docker:

- The **cold-start skip gate** (``_prewarmed_setup_satisfied`` applied at both
  ``_provision`` and ``_provision_run``): ``install_egress_ca`` /
  ``install_packages`` are skipped iff the image the container RAN FROM is a
  prewarm bake of the CURRENT base; the gate fails toward DOING the work on any
  mismatch/absence; and ``apply_network_lockdown`` (via ``_apply_egress_rules``)
  is NEVER skipped.
- The **operator bake command** (``bake_prewarm_image``): the committed image is
  stamped ``PREWARM_LABEL_KEY``+``BASE_IMAGE_LABEL_KEY`` and NOT the
  managed/instance/session labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.cli.commands.sandbox import bake_prewarm_image
from aios.models.environments import EnvironmentConfig
from aios.sandbox.backends.base import (
    BASE_IMAGE_LABEL_KEY,
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    PREWARM_LABEL_KEY,
    SESSION_LABEL_KEY,
    Mount,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.spec import ProvisioningPlan
from tests.helpers.sandbox import FakeBackend, limited_env

BASE = "ghcr.io/eumemic/aios-sandbox:latest"


def _make_spec(*, snapshot_image: str | None = None, image: str = BASE) -> SandboxSpec:
    return SandboxSpec(
        session_id="sess_01TEST",
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image=image,
        snapshot_image=snapshot_image,
    )


def _make_plan(
    spec: SandboxSpec, *, env_config: EnvironmentConfig | None = None
) -> ProvisioningPlan:
    return ProvisioningPlan(
        spec=spec,
        env_config=env_config,
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
        env_var_credentials=(),
    )


# ── shared fixtures for driving _provision / _provision_run ──────────────────


def _patch_provision_preamble(registry: SandboxRegistry, plan: ProvisioningPlan) -> tuple[Any, ...]:
    """Patch the session ``_provision`` preamble so it reaches the setup execs.

    ``_resolve_snapshot`` is patched to return ``plan.spec`` unchanged — the
    gate must read labels off ``spec.snapshot_image or spec.image``, which the
    resolved spec carries.
    """
    return (
        patch(
            "aios.sandbox.registry.build_spec_from_session",
            AsyncMock(return_value=plan),
        ),
        patch.object(registry, "_salvage_session_corpses", AsyncMock()),
        patch.object(registry, "_reconcile_pointer_from_local", AsyncMock()),
        patch.object(registry, "_resolve_snapshot", AsyncMock(return_value=plan.spec)),
    )


def _enter(patches: tuple[Any, ...]) -> list[Any]:
    ctxs = [p.__enter__() for p in patches]
    return ctxs


def _exit(patches: tuple[Any, ...]) -> None:
    for p in reversed(patches):
        p.__exit__(None, None, None)


class TestColdStartSkipGate:
    @pytest.mark.parametrize(
        ("labels_by_ref", "expect_skipped"),
        [
            # Prewarm of the current base → skip.
            ({BASE: {PREWARM_LABEL_KEY: BASE, BASE_IMAGE_LABEL_KEY: BASE}}, True),
            # Prewarm of an OLD base → do the work (base-drift discipline).
            ({BASE: {PREWARM_LABEL_KEY: "old-base", BASE_IMAGE_LABEL_KEY: "old-base"}}, False),
            # No prewarm label at all → do the work.
            ({BASE: {BASE_IMAGE_LABEL_KEY: BASE}}, False),
            # image_labels returns None (absent) → do the work.
            ({BASE: None}, False),
            # image_labels has no entry for the ref → do the work.
            ({}, False),
        ],
    )
    async def test_provision_skip_matrix(
        self, labels_by_ref: dict[str, dict[str, str] | None], expect_skipped: bool
    ) -> None:
        backend = FakeBackend(image_labels_by_ref=labels_by_ref)
        registry = SandboxRegistry(backend=backend)
        plan = _make_plan(_make_spec())

        ca = AsyncMock()
        pkgs = AsyncMock()
        patches = (
            *_patch_provision_preamble(registry, plan),
            patch("aios.sandbox.registry.install_egress_ca", ca),
            patch("aios.sandbox.registry.install_packages", pkgs),
            patch.object(registry, "_apply_egress_rules", AsyncMock()),
            patch("aios.sandbox.registry.log", MagicMock()),
        )
        _enter(patches)
        try:
            await registry._provision("sess_01TEST")
        finally:
            _exit(patches)

        if expect_skipped:
            ca.assert_not_awaited()
            pkgs.assert_not_awaited()
        else:
            ca.assert_awaited_once()
            pkgs.assert_awaited_once()

    async def test_resume_never_skips(self) -> None:
        """On a resume ``run_image`` is the tenant snapshot tag, whose labels
        lack ``PREWARM_LABEL_KEY`` — the execs MUST run even though the base
        image happens to be a prewarm of the current base."""
        snap = "aios-snap:sess_01TEST"
        backend = FakeBackend(
            image_labels_by_ref={
                # Base IS a prewarm — but we run from the snapshot, not the base.
                BASE: {PREWARM_LABEL_KEY: BASE, BASE_IMAGE_LABEL_KEY: BASE},
                snap: {BASE_IMAGE_LABEL_KEY: BASE},  # tenant snapshot: no prewarm label
            }
        )
        registry = SandboxRegistry(backend=backend)
        plan = _make_plan(_make_spec(snapshot_image=snap))

        ca = AsyncMock()
        pkgs = AsyncMock()
        patches = (
            *_patch_provision_preamble(registry, plan),
            patch("aios.sandbox.registry.install_egress_ca", ca),
            patch("aios.sandbox.registry.install_packages", pkgs),
            patch.object(registry, "_apply_egress_rules", AsyncMock()),
            patch("aios.sandbox.registry.log", MagicMock()),
        )
        _enter(patches)
        try:
            await registry._provision("sess_01TEST")
        finally:
            _exit(patches)

        ca.assert_awaited_once()
        pkgs.assert_awaited_once()

    @pytest.mark.parametrize("expect_skipped", [True, False])
    async def test_provision_run_parity(self, expect_skipped: bool) -> None:
        labels = (
            {PREWARM_LABEL_KEY: BASE, BASE_IMAGE_LABEL_KEY: BASE}
            if expect_skipped
            else {PREWARM_LABEL_KEY: "old-base"}
        )
        backend = FakeBackend(image_labels_by_ref={BASE: labels})
        registry = SandboxRegistry(backend=backend)
        # Runs never resolve a snapshot ⇒ snapshot_image is None ⇒ run from base.
        plan = _make_plan(_make_spec(snapshot_image=None))

        ca = AsyncMock()
        pkgs = AsyncMock()
        patches = (
            patch("aios.sandbox.registry.build_spec_from_run", AsyncMock(return_value=plan)),
            patch("aios.sandbox.registry.install_egress_ca", ca),
            patch("aios.sandbox.registry.install_packages", pkgs),
            patch.object(registry, "_apply_egress_rules", AsyncMock()),
            patch("aios.sandbox.registry.log", MagicMock()),
        )
        _enter(patches)
        try:
            await registry._provision_run("wfr_01TEST")
        finally:
            _exit(patches)

        if expect_skipped:
            ca.assert_not_awaited()
            pkgs.assert_not_awaited()
        else:
            ca.assert_awaited_once()
            pkgs.assert_awaited_once()

    async def test_lockdown_never_skipped_even_when_prewarmed(self) -> None:
        """Load-bearing negative: a Limited session against a prewarmed base
        still runs ``apply_network_lockdown`` (and thus ``run_netns_sidecar``).
        The skip gates only the CA/package execs, NOT ``_apply_egress_rules``."""
        backend = FakeBackend(
            image_labels_by_ref={BASE: {PREWARM_LABEL_KEY: BASE, BASE_IMAGE_LABEL_KEY: BASE}}
        )
        registry = SandboxRegistry(backend=backend)
        plan = _make_plan(_make_spec(), env_config=limited_env("pypi.org"))

        lockdown = AsyncMock()
        tool_broker = MagicMock()
        tool_broker.port = 54321
        runtime = MagicMock()
        runtime.require_tool_broker = MagicMock(return_value=tool_broker)

        patches = (
            *_patch_provision_preamble(registry, plan),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", lockdown),
            patch("aios.harness.runtime.require_tool_broker", runtime.require_tool_broker),
            patch("aios.sandbox.registry.log", MagicMock()),
        )
        _enter(patches)
        try:
            await registry._provision("sess_01TEST")
        finally:
            _exit(patches)

        lockdown.assert_awaited_once()


class TestPrewarmBakeCommand:
    async def test_bake_stamps_prewarm_and_base_labels_only(self) -> None:
        backend = FakeBackend()
        with (
            patch("aios.cli.commands.sandbox.install_egress_ca", AsyncMock()) as ca,
            patch("aios.cli.commands.sandbox.install_packages", AsyncMock()) as pkgs,
        ):
            tag = await bake_prewarm_image(backend, base_image=BASE, tag="aios-prewarm:latest")

        assert tag == "aios-prewarm:latest"
        # CA installed; no --environment ⇒ packages not installed.
        ca.assert_awaited_once()
        pkgs.assert_not_awaited()

        # Exactly one commit, stamped with both prewarm labels = base ref, and
        # NONE of the managed/instance/session labels.
        assert len(backend.prewarm_commits) == 1
        committed_tag, labels = backend.prewarm_commits[0]
        assert committed_tag == "aios-prewarm:latest"
        assert labels[PREWARM_LABEL_KEY] == BASE
        assert labels[BASE_IMAGE_LABEL_KEY] == BASE
        for forbidden in (MANAGED_LABEL_KEY, INSTANCE_LABEL_KEY, SESSION_LABEL_KEY):
            assert forbidden not in labels

        # Plain run of the base, and the transient container is torn down.
        verbs = [c[0] for c in backend.calls]
        assert "prewarm_run" in verbs
        assert "prewarm_remove" in verbs

    async def test_bake_with_environment_installs_packages(self) -> None:
        backend = FakeBackend()
        env_config = limited_env("pypi.org")
        with (
            patch("aios.cli.commands.sandbox.install_egress_ca", AsyncMock()),
            patch("aios.cli.commands.sandbox.install_packages", AsyncMock()) as pkgs,
            patch(
                "aios.cli.commands.sandbox._load_environment_config",
                AsyncMock(return_value=env_config),
            ),
        ):
            await bake_prewarm_image(
                backend,
                base_image=BASE,
                tag="aios-prewarm:env",
                environment_id="env_X",
                account_id="acct_X",
            )

        pkgs.assert_awaited_once()
        _tag, labels = backend.prewarm_commits[0]
        assert labels[PREWARM_LABEL_KEY] == BASE

    async def test_bake_removes_container_even_on_commit_failure(self) -> None:
        from aios.sandbox.backends.base import SandboxBackendError

        backend = FakeBackend()
        backend.prewarm_commit = AsyncMock(side_effect=SandboxBackendError("boom"))  # type: ignore[method-assign]
        with (
            patch("aios.cli.commands.sandbox.install_egress_ca", AsyncMock()),
            patch("aios.cli.commands.sandbox.install_packages", AsyncMock()),
            pytest.raises(SandboxBackendError),
        ):
            await bake_prewarm_image(backend, base_image=BASE, tag="aios-prewarm:latest")

        assert ("prewarm_remove", {"sandbox_id": "prewarm_container_id"}) in backend.calls
