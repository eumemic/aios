from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="BootstrapRequest")


@_attrs_define
class BootstrapRequest:
    """Body for ``POST /v1/accounts/bootstrap``.

    Only the human-readable display_name is required at bootstrap time;
    metadata can be added later via a PATCH endpoint (lands in PR 6).

        Attributes:
            display_name (str):
    """

    display_name: str

    def to_dict(self) -> dict[str, Any]:
        display_name = self.display_name

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "display_name": display_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        display_name = d.pop("display_name")

        bootstrap_request = cls(
            display_name=display_name,
        )

        return bootstrap_request
