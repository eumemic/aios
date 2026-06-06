from enum import Enum


class ListSessionsStatusType0(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"

    def __str__(self) -> str:
        return str(self.value)
