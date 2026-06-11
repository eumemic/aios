"""Pure-function tests for ``mount_snapshot_from_echoes`` env-var fold (#877).

The mount snapshot is computed at TWO sites with IDENTICAL membership:
``_assemble_plan`` (provision time, folding ``ResolvedEnvVarCredential``)
and ``release_if_mounts_changed`` (step time, folding
``EnvVarCredentialEcho``). If the two folds diverged, every session with
env-var creds would spuriously recycle on its first step. These tests pin
the fold as a pure function of ``(credential_id, updated_at)`` and prove
the structural symmetry between the two input shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aios.db.queries import EnvVarCredentialEcho
from aios.ids import VAULT_CREDENTIAL
from aios.sandbox.spec import mount_snapshot_from_echoes
from aios.services.vaults import ResolvedEnvVarCredential

_TS_V1 = datetime(2026, 6, 10, tzinfo=UTC)
_TS_V2 = datetime(2026, 6, 11, tzinfo=UTC)


def _resolved(credential_id: str, updated_at: datetime) -> ResolvedEnvVarCredential:
    return ResolvedEnvVarCredential(
        credential_id=credential_id,
        secret_name="GITHUB_TOKEN",
        secret_value="SENTINEL_DO_NOT_LEAK",
        allowed_hosts=("api.github.com",),
        updated_at=updated_at,
        placeholder="AIOS_SECRET_PLACEHOLDER_" + "ab" * 16,
    )


def _echo(credential_id: str, updated_at: datetime) -> EnvVarCredentialEcho:
    return EnvVarCredentialEcho(credential_id=credential_id, updated_at=updated_at)


def test_env_var_cred_folds_into_snapshot() -> None:
    """An env-var cred contributes exactly ``(VAULT_CREDENTIAL, id, ts)``."""
    snapshot = mount_snapshot_from_echoes([], [], [_resolved("vcr_01", _TS_V1)])
    assert (VAULT_CREDENTIAL, "vcr_01", _TS_V1.isoformat()) in snapshot


def test_rotation_changes_snapshot() -> None:
    """Same credential_id, bumped updated_at ⇒ different frozenset (a rotation
    propagates to a recycle, mirroring the github ``updated_at`` fold)."""
    before = mount_snapshot_from_echoes([], [], [_resolved("vcr_01", _TS_V1)])
    after = mount_snapshot_from_echoes([], [], [_resolved("vcr_01", _TS_V2)])
    assert before != after


def test_adding_a_cred_changes_snapshot() -> None:
    before = mount_snapshot_from_echoes([], [], [_resolved("vcr_01", _TS_V1)])
    after = mount_snapshot_from_echoes(
        [], [], [_resolved("vcr_01", _TS_V1), _resolved("vcr_02", _TS_V1)]
    )
    assert before != after


def test_removing_a_cred_changes_snapshot() -> None:
    """Archiving/removing a cred drops its echo ⇒ different frozenset."""
    before = mount_snapshot_from_echoes(
        [], [], [_resolved("vcr_01", _TS_V1), _resolved("vcr_02", _TS_V1)]
    )
    after = mount_snapshot_from_echoes([], [], [_resolved("vcr_01", _TS_V1)])
    assert before != after


def test_reorder_is_order_independent() -> None:
    """The fold is set membership — reordering the cred list is a no-op."""
    a = mount_snapshot_from_echoes(
        [], [], [_resolved("vcr_01", _TS_V1), _resolved("vcr_02", _TS_V2)]
    )
    b = mount_snapshot_from_echoes(
        [], [], [_resolved("vcr_02", _TS_V2), _resolved("vcr_01", _TS_V1)]
    )
    assert a == b


def test_resolved_and_echo_fold_identically() -> None:
    """Constraint A: a ``ResolvedEnvVarCredential`` (provision) and an
    ``EnvVarCredentialEcho`` (step) with the SAME ``(credential_id,
    updated_at)`` produce the SAME frozenset element.

    If they diverged, a session would recycle on its first step because the
    provision-time snapshot stamped on the handle would never match the
    step-time echo set.
    """
    provision = mount_snapshot_from_echoes([], [], [_resolved("vcr_01", _TS_V1)])
    step = mount_snapshot_from_echoes([], [], [_echo("vcr_01", _TS_V1)])
    assert provision == step
