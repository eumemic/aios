from enum import Enum


class TraceEntryKind(str, Enum):
    AGENT_CALL = "agent_call"
    ANNOTATION = "annotation"
    ERROR = "error"
    GATE = "gate"
    MESSAGE = "message"
    REQUEST = "request"
    RESPONSE = "response"
    RUN = "run"
    SESSION = "session"
    TOOL_CALL = "tool_call"

    def __str__(self) -> str:
        return str(self.value)
