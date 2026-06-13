from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    IDLE = "idle"

    def __str__(self) -> str:
        return str(self.value)
