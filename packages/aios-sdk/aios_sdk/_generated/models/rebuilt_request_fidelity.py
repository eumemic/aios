from enum import Enum


class RebuiltRequestFidelity(str, Enum):
    EXACT = "exact"
    INEXACT = "inexact"
    RERENDERED = "rerendered"

    def __str__(self) -> str:
        return str(self.value)
