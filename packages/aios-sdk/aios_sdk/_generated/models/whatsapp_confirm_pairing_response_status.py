from enum import Enum


class WhatsappConfirmPairingResponseStatus(str, Enum):
    ERROR = "error"
    SUCCESS = "success"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return str(self.value)
