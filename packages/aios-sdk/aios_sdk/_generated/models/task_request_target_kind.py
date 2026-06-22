from enum import Enum


class TaskRequestTargetKind(str, Enum):
    AGENT = "agent"
    SESSION = "session"
    WORKFLOW = "workflow"

    def __str__(self) -> str:
        return str(self.value)
