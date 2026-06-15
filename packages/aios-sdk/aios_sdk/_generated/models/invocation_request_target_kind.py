from enum import Enum


class InvocationRequestTargetKind(str, Enum):
    AGENT = "agent"
    SESSION = "session"
    WORKFLOW = "workflow"

    def __str__(self) -> str:
        return str(self.value)
