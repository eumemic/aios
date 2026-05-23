from enum import Enum


class McpToolsetConfigTransportType0(str, Enum):
    AGENT_TOOL = "agent_tool"
    BOTH = "both"
    CLI = "cli"

    def __str__(self) -> str:
        return str(self.value)
