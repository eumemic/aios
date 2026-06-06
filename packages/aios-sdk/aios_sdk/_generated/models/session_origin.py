from enum import Enum


class SessionOrigin(str, Enum):
    BACKGROUND = "background"
    FOREGROUND = "foreground"

    def __str__(self) -> str:
        return str(self.value)
