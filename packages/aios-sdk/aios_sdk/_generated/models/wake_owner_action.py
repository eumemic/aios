from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="WakeOwnerAction")


@_attrs_define
class WakeOwnerAction:
    """Deliver ``content`` as a user-role message to the trigger's OWNING
    session, waking the model — the self-wake primitive on a timer.

    No ``session_id``: the target is implicitly ``owner_session_id`` (NOT
    NULL this slice). An explicit-target wake is a separate future kind,
    not a field on this one (avoids the uncapped cross-session wake that an
    unconstrained ``session_id`` would be — see the ``wake_session`` tool's
    depth/rate caps).

        Attributes:
            content (str):
            kind (Literal['wake_owner'] | Unset):  Default: 'wake_owner'.
    """

    content: str
    kind: Literal["wake_owner"] | Unset = "wake_owner"

    def to_dict(self) -> dict[str, Any]:
        content = self.content

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "content": content,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        content = d.pop("content")

        kind = cast(Literal["wake_owner"] | Unset, d.pop("kind", UNSET))
        if kind != "wake_owner" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'wake_owner', got '{kind}'")

        wake_owner_action = cls(
            content=content,
            kind=kind,
        )

        return wake_owner_action
