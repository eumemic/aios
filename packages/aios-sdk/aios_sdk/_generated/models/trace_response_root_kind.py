from enum import Enum


class TraceResponseRootKind(str, Enum):
    RUN = "run"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
