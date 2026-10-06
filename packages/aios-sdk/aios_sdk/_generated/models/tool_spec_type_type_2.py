from enum import Enum


class ToolSpecTypeType2(str, Enum):
    GET_REQUEST = "get_request"
    SAMPLE_REQUESTS = "sample_requests"

    def __str__(self) -> str:
        return str(self.value)
