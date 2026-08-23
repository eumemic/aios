from enum import Enum


class UsageConsumersResponseMetric(str, Enum):
    COST_MICROUSD = "cost_microusd"
    TOTAL_TOKENS = "total_tokens"

    def __str__(self) -> str:
        return str(self.value)
