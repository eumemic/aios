from enum import Enum


class TriggerCreatedLastFireStatusType0(str, Enum):
    ERROR = "error"
    OK = "ok"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return str(self.value)
