from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db.queries import vaults


@pytest.mark.asyncio
async def test_session_credential_collision_warns_with_winner_and_shadowed_matches() -> None:
    conn = MagicMock()
    conn.fetch = AsyncMock(
        return_value=[
            {
                "ciphertext": b"winner-ciphertext",
                "nonce": b"winner-nonce",
                "auth_type": "bearer_header",
                "vault_id": "vlt_winner",
                "credential_id": "vcr_winner",
                "rank": 0,
            },
            {
                "ciphertext": b"shadowed-ciphertext",
                "nonce": b"shadowed-nonce",
                "auth_type": "bearer_header",
                "vault_id": "vlt_shadowed",
                "credential_id": "vcr_shadowed",
                "rank": 1,
            },
        ]
    )

    with patch.object(vaults, "log") as log:
        result = await vaults.resolve_session_credential(
            conn,
            "sess_test",
            "https://api.github.com",
            account_id="acc_test",
        )

    assert result is not None
    assert result[2] == "vlt_winner"
    log.warning.assert_called_once_with(
        "vault.credential_collision",
        session_id="sess_test",
        target_url="https://api.github.com",
        winning_credential_id="vcr_winner",
        winning_rank=0,
        shadowed_credentials=[{"credential_id": "vcr_shadowed", "rank": 1}],
    )


@pytest.mark.asyncio
async def test_run_credential_collision_warns_with_winner_and_shadowed_matches() -> None:
    conn = MagicMock()
    conn.fetch = AsyncMock(
        return_value=[
            {
                "ciphertext": b"winner-ciphertext",
                "nonce": b"winner-nonce",
                "auth_type": "bearer_header",
                "vault_id": "vlt_winner",
                "credential_id": "vcr_winner",
                "rank": 2,
            },
            {
                "ciphertext": b"shadowed-ciphertext",
                "nonce": b"shadowed-nonce",
                "auth_type": "basic",
                "vault_id": "vlt_shadowed",
                "credential_id": "vcr_shadowed",
                "rank": 4,
            },
        ]
    )

    with patch.object(vaults, "log") as log:
        result = await vaults.resolve_run_credential(
            conn,
            "wfr_test",
            "https://api.example.com",
            account_id="acc_test",
        )

    assert result is not None
    assert result[2] == "vlt_winner"
    log.warning.assert_called_once_with(
        "vault.credential_collision",
        run_id="wfr_test",
        target_url="https://api.example.com",
        winning_credential_id="vcr_winner",
        winning_rank=2,
        shadowed_credentials=[{"credential_id": "vcr_shadowed", "rank": 4}],
    )
