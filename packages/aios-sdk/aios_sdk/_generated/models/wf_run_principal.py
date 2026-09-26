from enum import Enum


class WfRunPrincipal(str, Enum):
    OPERATOR = "operator"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
