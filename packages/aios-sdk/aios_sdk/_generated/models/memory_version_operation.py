from enum import Enum


class MemoryVersionOperation(str, Enum):
    CREATED = "created"
    DELETED = "deleted"
    MODIFIED = "modified"

    def __str__(self) -> str:
        return str(self.value)
