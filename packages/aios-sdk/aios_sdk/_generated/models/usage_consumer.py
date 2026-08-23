from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.usage_consumer_kind import UsageConsumerKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.attributed_usage import AttributedUsage


T = TypeVar("T", bound="UsageConsumer")


@_attrs_define
class UsageConsumer:
    """One root consumer in the ranked account-wide usage view.

    Attributes:
        rank (int):
        kind (UsageConsumerKind):
        id (str):
        label (str):
        status (str):
        created_at (datetime.datetime):
        share (float):
        usage (AttributedUsage): Own and transitive usage for one node in the accounting tree.
        archived_at (datetime.datetime | None | Unset):
    """

    rank: int
    kind: UsageConsumerKind
    id: str
    label: str
    status: str
    created_at: datetime.datetime
    share: float
    usage: AttributedUsage
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rank = self.rank

        kind = self.kind.value

        id = self.id

        label = self.label

        status = self.status

        created_at = self.created_at.isoformat()

        share = self.share

        usage = self.usage.to_dict()

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "rank": rank,
                "kind": kind,
                "id": id,
                "label": label,
                "status": status,
                "created_at": created_at,
                "share": share,
                "usage": usage,
            }
        )
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.attributed_usage import AttributedUsage

        d = dict(src_dict)
        rank = d.pop("rank")

        kind = UsageConsumerKind(d.pop("kind"))

        id = d.pop("id")

        label = d.pop("label")

        status = d.pop("status")

        created_at = isoparse(d.pop("created_at"))

        share = d.pop("share")

        usage = AttributedUsage.from_dict(d.pop("usage"))

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        usage_consumer = cls(
            rank=rank,
            kind=kind,
            id=id,
            label=label,
            status=status,
            created_at=created_at,
            share=share,
            usage=usage,
            archived_at=archived_at,
        )

        usage_consumer.additional_properties = d
        return usage_consumer

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
