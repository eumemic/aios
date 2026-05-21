from enum import Enum


class WaitResponseSessionStatus(str, Enum):
    ERRORED = "errored"
    IDLE = "idle"
    PENDING = "pending"
    RESCHEDULING = "rescheduling"
    TERMINATED = "terminated"

    def __str__(self) -> str:
        return str(self.value)
