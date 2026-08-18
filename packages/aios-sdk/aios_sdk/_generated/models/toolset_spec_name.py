from enum import Enum


class ToolsetSpecName(str, Enum):
    AGENT_MANAGEMENT = "agent_management"
    DELEGATION = "delegation"
    GOAL_MANAGEMENT = "goal_management"
    TRIGGER_MANAGEMENT = "trigger_management"
    WORKFLOW_MANAGEMENT = "workflow_management"

    def __str__(self) -> str:
        return str(self.value)
