from enum import Enum


class ToolSpecTypeType0(str, Enum):
    BASH = "bash"
    CANCEL = "cancel"
    EDIT = "edit"
    GLOB = "glob"
    GREP = "grep"
    HTTP_REQUEST = "http_request"
    READ = "read"
    SCHEDULE_WAKE = "schedule_wake"
    SEARCH_EVENTS = "search_events"
    WAKE_SESSION = "wake_session"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    WRITE = "write"

    def __str__(self) -> str:
        return str(self.value)
