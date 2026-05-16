from enum import Enum


class VaultCredentialCreateAuthType(str, Enum):
    BASIC = "basic"
    BEARER_HEADER = "bearer_header"
    OAUTH2_REFRESH = "oauth2_refresh"

    def __str__(self) -> str:
        return str(self.value)
