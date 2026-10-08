from enum import Enum


class WfRunPrincipal(str, Enum):
    AGENT = "agent"
    OPERATOR = "operator"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
