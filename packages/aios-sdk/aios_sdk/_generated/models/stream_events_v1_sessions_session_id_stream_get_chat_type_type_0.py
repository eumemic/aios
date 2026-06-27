from enum import Enum


class StreamEventsV1SessionsSessionIdStreamGetChatTypeType0(str, Enum):
    DM = "dm"
    GROUP = "group"

    def __str__(self) -> str:
        return str(self.value)
