"""Unit tests for session-scoped vault credential self-visibility (#1945)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers builtins
from aios.tools.invoke import ToolBail, invoke_builtin

_SESSION = "ses_self"
_ACCOUNT = "acc_self"
_NOW = datetime(2026, 8, 13, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _stub_identity(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )


class _Conn:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: Any) -> None:
        return None


def _stub_runtime(monkeypatch: Any, rows: list[Any]) -> AsyncMock:
    query = AsyncMock(return_value=rows)
    monkeypatch.setattr("aios.db.queries.list_session_vault_credentials", query)
    monkeypatch.setattr(
        "aios.harness.runtime.require_pool", lambda: SimpleNamespace(acquire=lambda: _Conn())
    )
    return query


async def test_lists_only_safe_metadata_for_executing_session(monkeypatch: Any) -> None:
    from aios.db.queries.vaults import SessionVaultCredentialMetadata

    rows = [
        SessionVaultCredentialMetadata(
            credential_id="cred_mailgun",
            vault_id="vault_ops",
            display_name="Mailgun",
            auth_type="environment_variable",
            secret_name="MAILGUN_API_KEY",
            allowed_hosts=("api.mailgun.net",),
            target_url=None,
            created_at=_NOW,
            archived_at=None,
        ),
        SessionVaultCredentialMetadata(
            credential_id="cred_old",
            vault_id="vault_ops",
            display_name="Old GitHub",
            auth_type="environment_variable",
            secret_name="GITHUB_TOKEN",
            allowed_hosts=("github.com", "api.github.com"),
            target_url=None,
            created_at=_NOW,
            archived_at=_NOW,
        ),
    ]
    query = _stub_runtime(monkeypatch, rows)

    out = await invoke_builtin(_SESSION, "list_vault_credentials", {})

    assert out == {
        "credentials": [
            {
                "credential_id": "cred_mailgun",
                "vault_id": "vault_ops",
                "display_name": "Mailgun",
                "auth_type": "environment_variable",
                "secret_name": "MAILGUN_API_KEY",
                "allowed_hosts": ["api.mailgun.net"],
                "target_url": None,
                "created_at": "2026-08-13T00:00:00Z",
                "archived_at": None,
            },
            {
                "credential_id": "cred_old",
                "vault_id": "vault_ops",
                "display_name": "Old GitHub",
                "auth_type": "environment_variable",
                "secret_name": "GITHUB_TOKEN",
                "allowed_hosts": ["github.com", "api.github.com"],
                "target_url": None,
                "created_at": "2026-08-13T00:00:00Z",
                "archived_at": "2026-08-13T00:00:00Z",
            },
        ]
    }
    query.assert_awaited_once_with(ANY, _SESSION, account_id=_ACCOUNT)
    assert set(out["credentials"][0]) == {
        "credential_id",
        "vault_id",
        "display_name",
        "auth_type",
        "secret_name",
        "allowed_hosts",
        "target_url",
        "created_at",
        "archived_at",
    }


async def test_rejects_smuggled_session_or_account_identity() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(_SESSION, "list_vault_credentials", {"session_id": "ses_other"})


def test_registered_model_only() -> None:
    from aios.tools.registry import registry

    spec = registry.get("list_vault_credentials")
    assert spec.transport == "agent_tool"
    assert spec.parameters_schema["additionalProperties"] is False
