from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"

    def __str__(self) -> str:
        return str(self.value)
