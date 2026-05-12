from enum import Enum


class SessionStatus(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    RESCHEDULING = "rescheduling"
    RUNNING = "running"
    TERMINATED = "terminated"

    def __str__(self) -> str:
        return str(self.value)
