"""Regression coverage for vault credential metadata read semantics (#2135)."""

from datetime import UTC, datetime

from aios.db.queries.vaults import _row_to_vault_credential


def _credential_row(metadata: object) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "id": "vcr_test",
        "vault_id": "vlt_test",
        "display_name": "test",
        "target_url": "https://example.com",
        "auth_type": "bearer_header",
        "secret_name": None,
        "allowed_hosts": None,
        "metadata": metadata,
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
    }


def test_cleared_credential_metadata_round_trips_as_none() -> None:
    """The jsonb decoder returns Python None for a stored JSON null."""
    credential = _row_to_vault_credential(_credential_row(None))

    assert credential.metadata is None


def test_credential_metadata_round_trips_unchanged() -> None:
    metadata = {"owner": "platform", "nested": {"enabled": True}}

    credential = _row_to_vault_credential(_credential_row(metadata))

    assert credential.metadata == metadata
