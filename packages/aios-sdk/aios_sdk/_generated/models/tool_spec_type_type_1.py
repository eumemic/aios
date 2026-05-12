from enum import Enum


class ToolSpecTypeType1(str, Enum):
    CUSTOM = "custom"
    MCP_TOOLSET = "mcp_toolset"

    def __str__(self) -> str:
        return str(self.value)
