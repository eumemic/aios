from enum import Enum


class ListConnectionsModeType0(str, Enum):
    DETACHED = "detached"
    PER_CHAT = "per_chat"
    SINGLE_SESSION = "single_session"

    def __str__(self) -> str:
        return str(self.value)
