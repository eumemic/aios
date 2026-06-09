from enum import Enum


class WfRunStatus(str, Enum):
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERRORED = "errored"
    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"

    def __str__(self) -> str:
        return str(self.value)
