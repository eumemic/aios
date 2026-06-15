from enum import Enum


class WfRunEventType(str, Enum):
    ANNOTATION = "annotation"
    CALL_RESULT = "call_result"
    CALL_STARTED = "call_started"
    FRONTIER_DEFERRED = "frontier_deferred"
    REQUEST_RESPONSE = "request_response"
    RUN_COMPLETED = "run_completed"
    RUN_STARTED = "run_started"

    def __str__(self) -> str:
        return str(self.value)
