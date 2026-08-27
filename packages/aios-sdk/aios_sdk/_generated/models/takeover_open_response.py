from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.takeover_open_response_target import TakeoverOpenResponseTarget


T = TypeVar("T", bound="TakeoverOpenResponse")


@_attrs_define
class TakeoverOpenResponse:
    """The opened grant. The viewer PINS ``target``/``boot``/``epoch`` and
    refuses frames or input that do not match (the trusted-chrome binding,
    jarbot#106 §5.6).

        Attributes:
            grant_id (str):
            target (TakeoverOpenResponseTarget):
            boot (str):
            epoch (int):
            ttl_seconds (int):
    """

    grant_id: str
    target: TakeoverOpenResponseTarget
    boot: str
    epoch: int
    ttl_seconds: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        grant_id = self.grant_id

        target = self.target.to_dict()

        boot = self.boot

        epoch = self.epoch

        ttl_seconds = self.ttl_seconds

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "grant_id": grant_id,
                "target": target,
                "boot": boot,
                "epoch": epoch,
                "ttl_seconds": ttl_seconds,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.takeover_open_response_target import TakeoverOpenResponseTarget

        d = dict(src_dict)
        grant_id = d.pop("grant_id")

        target = TakeoverOpenResponseTarget.from_dict(d.pop("target"))

        boot = d.pop("boot")

        epoch = d.pop("epoch")

        ttl_seconds = d.pop("ttl_seconds")

        takeover_open_response = cls(
            grant_id=grant_id,
            target=target,
            boot=boot,
            epoch=epoch,
            ttl_seconds=ttl_seconds,
        )

        takeover_open_response.additional_properties = d
        return takeover_open_response

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
