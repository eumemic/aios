"""Unit coverage for the snapshot ref mint + the cross-tenant prefix gate (§5.1, §5.8)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.environments import RESERVED_SANDBOX_IMAGE_PREFIX, EnvironmentConfig
from aios.sandbox.backends.docker import _is_registry_image
from aios.sandbox.spec import snapshot_tag


class TestSnapshotTag:
    def test_shape_and_lowercasing(self) -> None:
        # ULIDs are uppercase; the tag lowercases the session id (bijective).
        tag = snapshot_tag("default", "sess_01HXYZ")
        assert tag == "aios-sbx-default-sess_01hxyz:latest"

    def test_is_not_a_registry_image(self) -> None:
        """Single path component ⇒ ``_is_registry_image`` False ⇒ the
        ``--pull always`` logic never reaches the registry for a snapshot."""
        tag = snapshot_tag("default", "sess_01HXYZ")
        assert _is_registry_image(tag) is False

    def test_pure_function_of_deployment_and_session(self) -> None:
        """The ref is a pure function of (deployment, session) — NEVER of which
        worker committed it (§5.11 invariant 7), so a multi-host handoff can't
        change a session's ref."""
        assert snapshot_tag("deploy_a", "sess_x") == snapshot_tag("deploy_a", "sess_x")
        assert snapshot_tag("deploy_a", "sess_x") != snapshot_tag("deploy_b", "sess_x")

    def test_uses_reserved_prefix(self) -> None:
        assert snapshot_tag("default", "sess_x").startswith(RESERVED_SANDBOX_IMAGE_PREFIX)


class TestReservedImagePrefixValidator:
    """The pydantic 422 layer of the two-layer cross-tenant gate (§5.8)."""

    def test_rejects_reserved_prefix(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            EnvironmentConfig(image="aios-sbx-default-sess_victim:latest")

    def test_rejects_reserved_prefix_case_insensitive(self) -> None:
        """Docker lowercases image refs, so the gate is case-insensitive."""
        with pytest.raises(ValidationError, match="reserved"):
            EnvironmentConfig(image="AIOS-SBX-default-sess_victim:latest")

    def test_allows_normal_images(self) -> None:
        cfg = EnvironmentConfig(image="ghcr.io/eumemic/aios-sandbox:latest")
        assert cfg.image == "ghcr.io/eumemic/aios-sandbox:latest"

    def test_allows_none(self) -> None:
        assert EnvironmentConfig(image=None).image is None
