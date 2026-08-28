from enum import Enum


class McpPermissionPolicyType(str, Enum):
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_ASK = "always_ask"
    AUTO_REVIEW = "auto_review"

    def __str__(self) -> str:
        return str(self.value)
