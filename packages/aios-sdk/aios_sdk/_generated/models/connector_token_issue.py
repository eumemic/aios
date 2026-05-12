from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorTokenIssue")


@_attrs_define
class ConnectorTokenIssue:
    """Request body for ``POST /v1/connector-tokens``.

    Attributes:
        connection_id (str):
        label (None | str | Unset):
    """

    connection_id: str
    label: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connection_id": connection_id,
            }
        )
        if label is not UNSET:
            field_dict["label"] = label

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        connector_token_issue = cls(
            connection_id=connection_id,
            label=label,
        )

        return connector_token_issue
