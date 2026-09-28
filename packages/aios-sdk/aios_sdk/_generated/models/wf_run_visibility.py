from enum import Enum


class WfRunVisibility(str, Enum):
    ACCOUNT = "account"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
