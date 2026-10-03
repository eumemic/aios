from enum import Enum


class MissingRequestMissing(str, Enum):
    ATTACHMENT = "attachment"
    BLOB = "blob"

    def __str__(self) -> str:
        return str(self.value)
