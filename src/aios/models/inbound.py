"""Internal models for MCP inbound subscriptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from aios.crypto.vault import EncryptedBlob


@dataclass(frozen=True, slots=True)
class InboundSubscriptionSpec:
    """One session/account subscription the inbound process should maintain."""

    session_id: str
    mcp_server_name: str
    mcp_server_url: str
    vault_id: str
    vault_credential_id: str
    account_id: str
    auth_type: str
    blob: EncryptedBlob
    credential_updated_at: datetime

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.session_id,
            self.mcp_server_name,
            self.mcp_server_url,
            self.vault_credential_id,
            self.account_id,
            self.credential_updated_at.isoformat(),
        )
