from enum import Enum


class AgentVersionPreemptPolicy(str, Enum):
    PREEMPT = "preempt"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
