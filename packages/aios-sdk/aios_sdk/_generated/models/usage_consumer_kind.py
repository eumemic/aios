from enum import Enum


class UsageConsumerKind(str, Enum):
    RUN = "run"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
