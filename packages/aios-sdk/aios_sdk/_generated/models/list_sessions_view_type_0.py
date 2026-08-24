from enum import Enum


class ListSessionsViewType0(str, Enum):
    FULL = "full"
    LITE = "lite"

    def __str__(self) -> str:
        return str(self.value)
