from enum import Enum


class InboundGrantStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    REVOKED = "revoked"

    def __str__(self) -> str:
        return str(self.value)
