from enum import Enum


class ActorType(str, Enum):
    API_ACTOR = "api_actor"
    SESSION_ACTOR = "session_actor"

    def __str__(self) -> str:
        return str(self.value)
