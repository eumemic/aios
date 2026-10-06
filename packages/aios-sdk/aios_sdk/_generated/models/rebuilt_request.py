from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.rebuilt_request_fidelity import RebuiltRequestFidelity
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.rebuilt_request_record import RebuiltRequestRecord
    from ..models.rebuilt_request_request import RebuiltRequestRequest


T = TypeVar("T", bound="RebuiltRequest")


@_attrs_define
class RebuiltRequest:
    """A captured request, recomposed (#2471).

    ``fidelity`` is ``exact`` when the rebuild hashes to the captured
    ``payload_sha``: it is the request the session composed. ``inexact`` means
    the renderer changed since (compare ``record.render_version``) or an image the
    request inlined changed on disk. ``rerendered`` means it was rendered for
    another model (``?model=``), so there's no hash to compare. ``request`` is
    ``{messages, tools, params}``; ``params`` never includes an ``api_key``.

        Attributes:
            session_id (str):
            request_id (str):
            fidelity (RebuiltRequestFidelity):
            request (RebuiltRequestRequest):
            record (RebuiltRequestRecord):
            kind (Literal['rebuilt'] | Unset):  Default: 'rebuilt'.
    """

    session_id: str
    request_id: str
    fidelity: RebuiltRequestFidelity
    request: RebuiltRequestRequest
    record: RebuiltRequestRecord
    kind: Literal["rebuilt"] | Unset = "rebuilt"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        session_id = self.session_id

        request_id = self.request_id

        fidelity = self.fidelity.value

        request = self.request.to_dict()

        record = self.record.to_dict()

        kind = self.kind

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "session_id": session_id,
                "request_id": request_id,
                "fidelity": fidelity,
                "request": request,
                "record": record,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.rebuilt_request_record import RebuiltRequestRecord
        from ..models.rebuilt_request_request import RebuiltRequestRequest

        d = dict(src_dict)
        session_id = d.pop("session_id")

        request_id = d.pop("request_id")

        fidelity = RebuiltRequestFidelity(d.pop("fidelity"))

        request = RebuiltRequestRequest.from_dict(d.pop("request"))

        record = RebuiltRequestRecord.from_dict(d.pop("record"))

        kind = cast(Literal["rebuilt"] | Unset, d.pop("kind", UNSET))
        if kind != "rebuilt" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'rebuilt', got '{kind}'")

        rebuilt_request = cls(
            session_id=session_id,
            request_id=request_id,
            fidelity=fidelity,
            request=request,
            record=record,
            kind=kind,
        )

        rebuilt_request.additional_properties = d
        return rebuilt_request

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
