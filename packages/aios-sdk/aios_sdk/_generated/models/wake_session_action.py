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

T = TypeVar("T", bound="WakeSessionAction")


@_attrs_define
class WakeSessionAction:
    """Deliver ``content`` as a user-role message to an EXPLICITLY-NAMED
    same-account session, waking it. The cross-session twin of wake_owner:
    runs through ``services.wake.deliver_cross_session_wake`` with the FIRING
    TRIGGER as the lineage root, so the wake_session depth/per-pair-rate caps
    apply (a fire→wake→fire cascade terminates in bounded steps).

    ``target_session_id`` is NOT resolved in the model — it is account-data,
    checked at fire time (same as ``workflow_id``). A cross-account / archived
    / cap-breaching target surfaces as a fire ``error`` (no silent drop), not a
    write-time rejection. Single named target only: no topics, no
    competing-consumers, no fan-out (issue #1280).

        Attributes:
            target_session_id (str):
            content (str):
            kind (Literal['wake_session'] | Unset):  Default: 'wake_session'.
    """

    target_session_id: str
    content: str
    kind: Literal["wake_session"] | Unset = "wake_session"

    def to_dict(self) -> dict[str, Any]:
        target_session_id = self.target_session_id

        content = self.content

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "target_session_id": target_session_id,
                "content": content,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        target_session_id = d.pop("target_session_id")

        content = d.pop("content")

        kind = cast(Literal["wake_session"] | Unset, d.pop("kind", UNSET))
        if kind != "wake_session" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'wake_session', got '{kind}'")

        wake_session_action = cls(
            target_session_id=target_session_id,
            content=content,
            kind=kind,
        )

        return wake_session_action
