from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.usage_counters import UsageCounters
    from ..models.usage_rate import UsageRate


T = TypeVar("T", bound="AttributedUsage")


@_attrs_define
class AttributedUsage:
    """Own and transitive usage for one node in the accounting tree.

    Attributes:
        own (UsageCounters | Unset): Cumulative inference bought at one node or in one subtree.
        subtree (UsageCounters | Unset): Cumulative inference bought at one node or in one subtree.
        own_rate (None | Unset | UsageRate):
        subtree_rate (None | Unset | UsageRate):
    """

    own: UsageCounters | Unset = UNSET
    subtree: UsageCounters | Unset = UNSET
    own_rate: None | Unset | UsageRate = UNSET
    subtree_rate: None | Unset | UsageRate = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.usage_rate import UsageRate

        own: dict[str, Any] | Unset = UNSET
        if not isinstance(self.own, Unset):
            own = self.own.to_dict()

        subtree: dict[str, Any] | Unset = UNSET
        if not isinstance(self.subtree, Unset):
            subtree = self.subtree.to_dict()

        own_rate: dict[str, Any] | None | Unset
        if isinstance(self.own_rate, Unset):
            own_rate = UNSET
        elif isinstance(self.own_rate, UsageRate):
            own_rate = self.own_rate.to_dict()
        else:
            own_rate = self.own_rate

        subtree_rate: dict[str, Any] | None | Unset
        if isinstance(self.subtree_rate, Unset):
            subtree_rate = UNSET
        elif isinstance(self.subtree_rate, UsageRate):
            subtree_rate = self.subtree_rate.to_dict()
        else:
            subtree_rate = self.subtree_rate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if own is not UNSET:
            field_dict["own"] = own
        if subtree is not UNSET:
            field_dict["subtree"] = subtree
        if own_rate is not UNSET:
            field_dict["own_rate"] = own_rate
        if subtree_rate is not UNSET:
            field_dict["subtree_rate"] = subtree_rate

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.usage_counters import UsageCounters
        from ..models.usage_rate import UsageRate

        d = dict(src_dict)
        _own = d.pop("own", UNSET)
        own: UsageCounters | Unset
        if isinstance(_own, Unset):
            own = UNSET
        else:
            own = UsageCounters.from_dict(_own)

        _subtree = d.pop("subtree", UNSET)
        subtree: UsageCounters | Unset
        if isinstance(_subtree, Unset):
            subtree = UNSET
        else:
            subtree = UsageCounters.from_dict(_subtree)

        def _parse_own_rate(data: object) -> None | Unset | UsageRate:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                own_rate_type_0 = UsageRate.from_dict(data)

                return own_rate_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UsageRate, data)

        own_rate = _parse_own_rate(d.pop("own_rate", UNSET))

        def _parse_subtree_rate(data: object) -> None | Unset | UsageRate:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                subtree_rate_type_0 = UsageRate.from_dict(data)

                return subtree_rate_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UsageRate, data)

        subtree_rate = _parse_subtree_rate(d.pop("subtree_rate", UNSET))

        attributed_usage = cls(
            own=own,
            subtree=subtree,
            own_rate=own_rate,
            subtree_rate=subtree_rate,
        )

        attributed_usage.additional_properties = d
        return attributed_usage

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
