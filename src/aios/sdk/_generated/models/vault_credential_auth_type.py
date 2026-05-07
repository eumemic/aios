from enum import Enum


class VaultCredentialAuthType(str, Enum):
    MCP_OAUTH = "mcp_oauth"
    STATIC_BEARER = "static_bearer"

    def __str__(self) -> str:
        return str(self.value)
