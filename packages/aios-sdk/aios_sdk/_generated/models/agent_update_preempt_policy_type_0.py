from enum import Enum


class AgentUpdatePreemptPolicyType0(str, Enum):
    PREEMPT = "preempt"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
