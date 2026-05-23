from enum import Enum


class McpToolConfigTransportType0(str, Enum):
    AGENT_TOOL = "agent_tool"
    BOTH = "both"
    CLI = "cli"

    def __str__(self) -> str:
        return str(self.value)
