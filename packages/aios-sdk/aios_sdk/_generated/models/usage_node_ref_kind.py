from enum import Enum


class UsageNodeRefKind(str, Enum):
    RUN = "run"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
