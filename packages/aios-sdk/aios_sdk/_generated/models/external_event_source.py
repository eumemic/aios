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

T = TypeVar("T", bound="ExternalEventSource")


@_attrs_define
class ExternalEventSource:
    """Reactive source: fires from an authenticated inbound webhook ingress.

    No wire fields beyond ``kind`` — the inbound HTTP body IS the event, and
    the per-trigger ingest secret is server-minted (NOT a wire field; returned
    plaintext-once on create and stored only as a SHA-256 hash). Like
    ``run_completion`` this is unschedulable by the tick (``next_fire``
    permanently NULL); fires are dispatched from the ingress edge instead of
    a run-completion transaction. It carries no defaulted fields, so (unlike
    ``RunCompletionSource``) it needs no ``*Replace`` subclass.

        Attributes:
            kind (Literal['external_event'] | Unset):  Default: 'external_event'.
    """

    kind: Literal["external_event"] | Unset = "external_event"

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = cast(Literal["external_event"] | Unset, d.pop("kind", UNSET))
        if kind != "external_event" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'external_event', got '{kind}'")

        external_event_source = cls(
            kind=kind,
        )

        return external_event_source
