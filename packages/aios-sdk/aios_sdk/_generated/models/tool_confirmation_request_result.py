from enum import Enum


class ToolConfirmationRequestResult(str, Enum):
    ALLOW = "allow"
    DENY = "deny"

    def __str__(self) -> str:
        return str(self.value)
