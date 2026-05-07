from enum import Enum


class ListV1SessionsGetStatusType0(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    RESCHEDULING = "rescheduling"
    RUNNING = "running"
    TERMINATED = "terminated"

    def __str__(self) -> str:
        return str(self.value)
