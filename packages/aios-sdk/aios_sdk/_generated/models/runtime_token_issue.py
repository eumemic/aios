from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="RuntimeTokenIssue")


@_attrs_define
class RuntimeTokenIssue:
    """Request body for ``POST /v1/runtime-tokens``.

    Attributes:
        connector (str):
        label (None | str | Unset):
        connection_ids (list[str] | None | Unset): Optional allowlist of connection IDs the issued token is authorized
            for.  Omit (``None``) to leave the token unscoped — it sees every connection of ``connector`` type.
    """

    connector: str
    label: None | str | Unset = UNSET
    connection_ids: list[str] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        connector = self.connector

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        connection_ids: list[str] | None | Unset
        if isinstance(self.connection_ids, Unset):
            connection_ids = UNSET
        elif isinstance(self.connection_ids, list):
            connection_ids = self.connection_ids

        else:
            connection_ids = self.connection_ids

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connector": connector,
            }
        )
        if label is not UNSET:
            field_dict["label"] = label
        if connection_ids is not UNSET:
            field_dict["connection_ids"] = connection_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        connector = d.pop("connector")

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        def _parse_connection_ids(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                connection_ids_type_0 = cast(list[str], data)

                return connection_ids_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        connection_ids = _parse_connection_ids(d.pop("connection_ids", UNSET))

        runtime_token_issue = cls(
            connector=connector,
            label=label,
            connection_ids=connection_ids,
        )

        return runtime_token_issue
