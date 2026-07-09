from enum import Enum


class AgentCreatePreemptPolicy(str, Enum):
    PREEMPT = "preempt"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
