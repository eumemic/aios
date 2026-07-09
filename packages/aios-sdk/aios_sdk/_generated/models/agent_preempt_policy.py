from enum import Enum


class AgentPreemptPolicy(str, Enum):
    PREEMPT = "preempt"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
