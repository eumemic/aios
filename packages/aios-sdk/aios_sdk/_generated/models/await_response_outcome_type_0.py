from enum import Enum


class AwaitResponseOutcomeType0(str, Enum):
    CANCELLED = "cancelled"
    ERRORED = "errored"
    OK = "ok"

    def __str__(self) -> str:
        return str(self.value)
