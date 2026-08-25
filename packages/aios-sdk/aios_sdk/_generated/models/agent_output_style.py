from enum import Enum


class AgentOutputStyle(str, Enum):
    CONCISE = "concise"
    DEFAULT = "default"

    def __str__(self) -> str:
        return str(self.value)
