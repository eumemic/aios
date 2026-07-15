from enum import Enum


class WfRunCreateWorkspace(str, Enum):
    FRESH = "fresh"
    SHARED = "shared"

    def __str__(self) -> str:
        return str(self.value)
