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

from ..models.missing_request_missing import MissingRequestMissing
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.missing_request_record import MissingRequestRecord


T = TypeVar("T", bound="MissingRequest")


@_attrs_define
class MissingRequest:
    """A captured request that can't be rebuilt: ``blob`` (a captured part is gone)
    or ``attachment`` (an image file the request inlined is unreadable here).

        Attributes:
            session_id (str):
            request_id (str):
            missing (MissingRequestMissing):
            record (MissingRequestRecord):
            kind (Literal['missing'] | Unset):  Default: 'missing'.
    """

    session_id: str
    request_id: str
    missing: MissingRequestMissing
    record: MissingRequestRecord
    kind: Literal["missing"] | Unset = "missing"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        session_id = self.session_id

        request_id = self.request_id

        missing = self.missing.value

        record = self.record.to_dict()

        kind = self.kind

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "session_id": session_id,
                "request_id": request_id,
                "missing": missing,
                "record": record,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.missing_request_record import MissingRequestRecord

        d = dict(src_dict)
        session_id = d.pop("session_id")

        request_id = d.pop("request_id")

        missing = MissingRequestMissing(d.pop("missing"))

        record = MissingRequestRecord.from_dict(d.pop("record"))

        kind = cast(Literal["missing"] | Unset, d.pop("kind", UNSET))
        if kind != "missing" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'missing', got '{kind}'")

        missing_request = cls(
            session_id=session_id,
            request_id=request_id,
            missing=missing,
            record=record,
            kind=kind,
        )

        missing_request.additional_properties = d
        return missing_request

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
