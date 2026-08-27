from enum import Enum


class TakeoverCloseRequestOutcome(str, Enum):
    CANCELLED = "cancelled"
    DONE = "done"

    def __str__(self) -> str:
        return str(self.value)
