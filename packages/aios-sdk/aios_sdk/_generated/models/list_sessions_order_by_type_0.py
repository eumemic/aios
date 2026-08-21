from enum import Enum


class ListSessionsOrderByType0(str, Enum):
    CREATED_AT = "created_at"
    LAST_EVENT_AT = "last_event_at"
    UPDATED_AT = "updated_at"

    def __str__(self) -> str:
        return str(self.value)
