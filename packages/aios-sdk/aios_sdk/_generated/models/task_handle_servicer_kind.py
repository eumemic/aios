from enum import Enum


class TaskHandleServicerKind(str, Enum):
    RUN = "run"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
