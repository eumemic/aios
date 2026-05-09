from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_metadata import ConnectionMetadata
    from ..models.tool_spec import ToolSpec


T = TypeVar("T", bound="Connection")


@_attrs_define
class Connection:
    """Read view of a connection.

    Mode is implicit in the populated field:

    * ``session_id`` set → single_session
    * ``session_template_id`` set → per_chat
    * neither → detached

        Attributes:
            id (str):
            connector (str):
            account (str):
            metadata (ConnectionMetadata):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            session_id (None | str | Unset):
            session_template_id (None | str | Unset):
            tools (list[ToolSpec] | Unset):
            attached_at (datetime.datetime | None | Unset):
            archived_at (datetime.datetime | None | Unset):
    """

    id: str
    connector: str
    account: str
    metadata: ConnectionMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    session_id: None | str | Unset = UNSET
    session_template_id: None | str | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    attached_at: datetime.datetime | None | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        connector = self.connector

        account = self.account

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        session_id: None | str | Unset
        if isinstance(self.session_id, Unset):
            session_id = UNSET
        else:
            session_id = self.session_id

        session_template_id: None | str | Unset
        if isinstance(self.session_template_id, Unset):
            session_template_id = UNSET
        else:
            session_template_id = self.session_template_id

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        attached_at: None | str | Unset
        if isinstance(self.attached_at, Unset):
            attached_at = UNSET
        elif isinstance(self.attached_at, datetime.datetime):
            attached_at = self.attached_at.isoformat()
        else:
            attached_at = self.attached_at

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
                "id": id,
                "connector": connector,
                "account": account,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if session_template_id is not UNSET:
            field_dict["session_template_id"] = session_template_id
        if tools is not UNSET:
            field_dict["tools"] = tools
        if attached_at is not UNSET:
            field_dict["attached_at"] = attached_at
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connection_metadata import ConnectionMetadata
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        id = d.pop("id")

        connector = d.pop("connector")

        account = d.pop("account")

        metadata = ConnectionMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_session_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        session_id = _parse_session_id(d.pop("session_id", UNSET))

        def _parse_session_template_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        session_template_id = _parse_session_template_id(
            d.pop("session_template_id", UNSET)
        )

        _tools = d.pop("tools", UNSET)
        tools: list[ToolSpec] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = ToolSpec.from_dict(tools_item_data)

                tools.append(tools_item)

        def _parse_attached_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                attached_at_type_0 = isoparse(data)

                return attached_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        attached_at = _parse_attached_at(d.pop("attached_at", UNSET))

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

        connection = cls(
            id=id,
            connector=connector,
            account=account,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            session_id=session_id,
            session_template_id=session_template_id,
            tools=tools,
            attached_at=attached_at,
            archived_at=archived_at,
        )

        connection.additional_properties = d
        return connection

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
