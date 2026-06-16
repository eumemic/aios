from enum import Enum


class TraceEntryTerminalStateType0(str, Enum):
    CANCELLED = "cancelled"
    ERRORED = "errored"
    OK = "ok"
    RUNNING = "running"
    SUSPENDED = "suspended"

    def __str__(self) -> str:
        return str(self.value)
