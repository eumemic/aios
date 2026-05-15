from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="MintKeyRequest")


@_attrs_define
class MintKeyRequest:
    """
    Attributes:
        label (str):
    """

    label: str

    def to_dict(self) -> dict[str, Any]:
        label = self.label

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "label": label,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        label = d.pop("label")

        mint_key_request = cls(
            label=label,
        )

        return mint_key_request
