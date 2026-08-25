from enum import Enum


class AgentUpdateOutputStyleType0(str, Enum):
    CONCISE = "concise"
    DEFAULT = "default"

    def __str__(self) -> str:
        return str(self.value)
