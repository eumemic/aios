from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyPostConnectorInbound")


@_attrs_define
class BodyPostConnectorInbound:
    """
    Attributes:
        event_id (str): Client-supplied dedup key (ULID).
        chat_id (str):
        content (str):
        sender (None | str | Unset): JSON-encoded sender dict (e.g. {"display_name": "Alice"}).
        metadata (None | str | Unset): JSON-encoded connector metadata dict.
        timestamp (None | str | Unset): Optional ISO-8601 platform timestamp; stored in event metadata.
        attachments (list[str] | None | Unset): One file part per attachment; filename + content-type read from each.
    """

    event_id: str
    chat_id: str
    content: str
    sender: None | str | Unset = UNSET
    metadata: None | str | Unset = UNSET
    timestamp: None | str | Unset = UNSET
    attachments: list[str] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        event_id = self.event_id

        chat_id = self.chat_id

        content = self.content

        sender: None | str | Unset
        if isinstance(self.sender, Unset):
            sender = UNSET
        else:
            sender = self.sender

        metadata: None | str | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        else:
            metadata = self.metadata

        timestamp: None | str | Unset
        if isinstance(self.timestamp, Unset):
            timestamp = UNSET
        else:
            timestamp = self.timestamp

        attachments: list[str] | None | Unset
        if isinstance(self.attachments, Unset):
            attachments = UNSET
        elif isinstance(self.attachments, list):
            attachments = self.attachments

        else:
            attachments = self.attachments

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "event_id": event_id,
                "chat_id": chat_id,
                "content": content,
            }
        )
        if sender is not UNSET:
            field_dict["sender"] = sender
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if timestamp is not UNSET:
            field_dict["timestamp"] = timestamp
        if attachments is not UNSET:
            field_dict["attachments"] = attachments

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        files.append(("event_id", (None, str(self.event_id).encode(), "text/plain")))

        files.append(("chat_id", (None, str(self.chat_id).encode(), "text/plain")))

        files.append(("content", (None, str(self.content).encode(), "text/plain")))

        if not isinstance(self.sender, Unset):
            if isinstance(self.sender, str):
                files.append(
                    ("sender", (None, str(self.sender).encode(), "text/plain"))
                )
            else:
                files.append(
                    ("sender", (None, str(self.sender).encode(), "text/plain"))
                )

        if not isinstance(self.metadata, Unset):
            if isinstance(self.metadata, str):
                files.append(
                    ("metadata", (None, str(self.metadata).encode(), "text/plain"))
                )
            else:
                files.append(
                    ("metadata", (None, str(self.metadata).encode(), "text/plain"))
                )

        if not isinstance(self.timestamp, Unset):
            if isinstance(self.timestamp, str):
                files.append(
                    ("timestamp", (None, str(self.timestamp).encode(), "text/plain"))
                )
            else:
                files.append(
                    ("timestamp", (None, str(self.timestamp).encode(), "text/plain"))
                )

        if not isinstance(self.attachments, Unset):
            if isinstance(self.attachments, list):
                for attachments_type_0_item_element in self.attachments:
                    files.append(
                        (
                            "attachments",
                            (
                                None,
                                str(attachments_type_0_item_element).encode(),
                                "text/plain",
                            ),
                        )
                    )
            else:
                files.append(
                    (
                        "attachments",
                        (None, str(self.attachments).encode(), "text/plain"),
                    )
                )

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        event_id = d.pop("event_id")

        chat_id = d.pop("chat_id")

        content = d.pop("content")

        def _parse_sender(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        sender = _parse_sender(d.pop("sender", UNSET))

        def _parse_metadata(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        def _parse_timestamp(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        timestamp = _parse_timestamp(d.pop("timestamp", UNSET))

        def _parse_attachments(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                attachments_type_0 = cast(list[str], data)

                return attachments_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        attachments = _parse_attachments(d.pop("attachments", UNSET))

        body_post_connector_inbound = cls(
            event_id=event_id,
            chat_id=chat_id,
            content=content,
            sender=sender,
            metadata=metadata,
            timestamp=timestamp,
            attachments=attachments,
        )

        body_post_connector_inbound.additional_properties = d
        return body_post_connector_inbound

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
