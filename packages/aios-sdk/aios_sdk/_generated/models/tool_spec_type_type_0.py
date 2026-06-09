from enum import Enum


class ToolSpecTypeType0(str, Enum):
    AWAIT_RUN = "await_run"
    BASH = "bash"
    CANCEL = "cancel"
    CREATE_RUN = "create_run"
    CREATE_WORKFLOW = "create_workflow"
    EDIT = "edit"
    GLOB = "glob"
    GREP = "grep"
    HTTP_REQUEST = "http_request"
    READ = "read"
    SCHEDULE_TASK_ADD = "schedule_task_add"
    SCHEDULE_TASK_REMOVE = "schedule_task_remove"
    SCHEDULE_TASK_UPDATE = "schedule_task_update"
    SCHEDULE_WAKE = "schedule_wake"
    SEARCH_EVENTS = "search_events"
    UPDATE_WORKFLOW = "update_workflow"
    WAKE_SELF = "wake_self"
    WAKE_SESSION = "wake_session"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    WRITE = "write"

    def __str__(self) -> str:
        return str(self.value)
