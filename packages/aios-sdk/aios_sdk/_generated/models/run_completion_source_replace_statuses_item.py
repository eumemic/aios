from enum import Enum


class RunCompletionSourceReplaceStatusesItem(str, Enum):
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERRORED = "errored"

    def __str__(self) -> str:
        return str(self.value)
