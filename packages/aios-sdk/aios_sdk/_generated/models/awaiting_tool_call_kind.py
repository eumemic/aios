from enum import Enum


class AwaitingToolCallKind(str, Enum):
    BUILTIN = "builtin"
    CUSTOM = "custom"
    MCP = "mcp"

    def __str__(self) -> str:
        return str(self.value)
