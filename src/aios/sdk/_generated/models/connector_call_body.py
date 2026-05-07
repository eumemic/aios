from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_call_body_arguments import ConnectorCallBodyArguments
    from ..models.connector_call_body_meta_type_0 import ConnectorCallBodyMetaType0


T = TypeVar("T", bound="ConnectorCallBody")


@_attrs_define
class ConnectorCallBody:
    """
    Attributes:
        tool (str):
        arguments (ConnectorCallBodyArguments | Unset):
        meta (ConnectorCallBodyMetaType0 | None | Unset):
    """

    tool: str
    arguments: ConnectorCallBodyArguments | Unset = UNSET
    meta: ConnectorCallBodyMetaType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.connector_call_body_meta_type_0 import ConnectorCallBodyMetaType0

        tool = self.tool

        arguments: dict[str, Any] | Unset = UNSET
        if not isinstance(self.arguments, Unset):
            arguments = self.arguments.to_dict()

        meta: dict[str, Any] | None | Unset
        if isinstance(self.meta, Unset):
            meta = UNSET
        elif isinstance(self.meta, ConnectorCallBodyMetaType0):
            meta = self.meta.to_dict()
        else:
            meta = self.meta

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "tool": tool,
            }
        )
        if arguments is not UNSET:
            field_dict["arguments"] = arguments
        if meta is not UNSET:
            field_dict["meta"] = meta

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connector_call_body_arguments import ConnectorCallBodyArguments
        from ..models.connector_call_body_meta_type_0 import ConnectorCallBodyMetaType0

        d = dict(src_dict)
        tool = d.pop("tool")

        _arguments = d.pop("arguments", UNSET)
        arguments: ConnectorCallBodyArguments | Unset
        if isinstance(_arguments, Unset):
            arguments = UNSET
        else:
            arguments = ConnectorCallBodyArguments.from_dict(_arguments)

        def _parse_meta(data: object) -> ConnectorCallBodyMetaType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                meta_type_0 = ConnectorCallBodyMetaType0.from_dict(data)

                return meta_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ConnectorCallBodyMetaType0 | None | Unset, data)

        meta = _parse_meta(d.pop("meta", UNSET))

        connector_call_body = cls(
            tool=tool,
            arguments=arguments,
            meta=meta,
        )

        return connector_call_body
