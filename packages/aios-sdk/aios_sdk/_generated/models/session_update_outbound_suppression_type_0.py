from enum import Enum


class SessionUpdateOutboundSuppressionType0(str, Enum):
    OFF = "off"
    ON = "on"

    def __str__(self) -> str:
        return str(self.value)
