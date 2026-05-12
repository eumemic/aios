from enum import Enum


class ListSessionEventsKindType0(str, Enum):
    INTERRUPT = "interrupt"
    LIFECYCLE = "lifecycle"
    MESSAGE = "message"
    SPAN = "span"

    def __str__(self) -> str:
        return str(self.value)
