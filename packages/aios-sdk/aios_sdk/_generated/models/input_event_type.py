from enum import Enum


class InputEventType(str, Enum):
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    POINTER_DOWN = "pointer_down"
    POINTER_MOVE = "pointer_move"
    POINTER_UP = "pointer_up"
    TEXT = "text"
    WHEEL = "wheel"

    def __str__(self) -> str:
        return str(self.value)
