from enum import Enum


class RunCompletionSourceStatusesItem(str, Enum):
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERRORED = "errored"

    def __str__(self) -> str:
        return str(self.value)
