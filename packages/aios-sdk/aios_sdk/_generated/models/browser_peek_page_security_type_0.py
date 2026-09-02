from enum import Enum


class BrowserPeekPageSecurityType0(str, Enum):
    INSECURE = "insecure"
    SECURE = "secure"

    def __str__(self) -> str:
        return str(self.value)
