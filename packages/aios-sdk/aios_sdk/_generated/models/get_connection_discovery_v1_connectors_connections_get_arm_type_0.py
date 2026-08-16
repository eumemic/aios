from enum import Enum


class GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0(str, Enum):
    FRESH = "fresh"
    TAIL = "tail"

    def __str__(self) -> str:
        return str(self.value)
