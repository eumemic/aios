from enum import Enum


class SessionCreateOutboundSuppression(str, Enum):
    OFF = "off"
    ON = "on"

    def __str__(self) -> str:
        return str(self.value)
