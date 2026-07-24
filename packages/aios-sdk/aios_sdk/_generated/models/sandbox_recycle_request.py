from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="SandboxRecycleRequest")


@_attrs_define
class SandboxRecycleRequest:
    """Explicit acknowledgement for discarding sandbox-local writable state.

    Attributes:
        discard_unsalvaged (bool):
    """

    discard_unsalvaged: bool

    def to_dict(self) -> dict[str, Any]:
        discard_unsalvaged = self.discard_unsalvaged

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "discard_unsalvaged": discard_unsalvaged,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        discard_unsalvaged = d.pop("discard_unsalvaged")

        sandbox_recycle_request = cls(
            discard_unsalvaged=discard_unsalvaged,
        )

        return sandbox_recycle_request
