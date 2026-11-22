from enum import Enum


class ToolsetMemberDefaultsPermissionType0(str, Enum):
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_ASK = "always_ask"

    def __str__(self) -> str:
        return str(self.value)
