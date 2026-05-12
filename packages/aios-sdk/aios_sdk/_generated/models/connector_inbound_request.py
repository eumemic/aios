from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_inbound_request_attachments_type_0_item import (
        ConnectorInboundRequestAttachmentsType0Item,
    )
    from ..models.connector_inbound_request_metadata_type_0 import (
        ConnectorInboundRequestMetadataType0,
    )
    from ..models.connector_inbound_request_sender import ConnectorInboundRequestSender


T = TypeVar("T", bound="ConnectorInboundRequest")


@_attrs_define
class ConnectorInboundRequest:
    """Body for ``POST /v1/connectors/inbound``.

    Authenticated via ``ConnectorAuthDep`` so the connection_id is
    server-resolved from the bearer token — clients don't pick which
    connection their inbound lands on.

        Attributes:
            event_id (str): Client-supplied dedup key (ULID).  Replays return the original event id.
            chat_id (str):
            content (str):
            sender (ConnectorInboundRequestSender | Unset):
            attachments (list[ConnectorInboundRequestAttachmentsType0Item] | None | Unset):
            metadata (ConnectorInboundRequestMetadataType0 | None | Unset):
            timestamp (None | str | Unset): Optional ISO-8601 platform timestamp; stored in event metadata.
    """

    event_id: str
    chat_id: str
    content: str
    sender: ConnectorInboundRequestSender | Unset = UNSET
    attachments: list[ConnectorInboundRequestAttachmentsType0Item] | None | Unset = (
        UNSET
    )
    metadata: ConnectorInboundRequestMetadataType0 | None | Unset = UNSET
    timestamp: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.connector_inbound_request_metadata_type_0 import (
            ConnectorInboundRequestMetadataType0,
        )

        event_id = self.event_id

        chat_id = self.chat_id

        content = self.content

        sender: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sender, Unset):
            sender = self.sender.to_dict()

        attachments: list[dict[str, Any]] | None | Unset
        if isinstance(self.attachments, Unset):
            attachments = UNSET
        elif isinstance(self.attachments, list):
            attachments = []
            for attachments_type_0_item_data in self.attachments:
                attachments_type_0_item = attachments_type_0_item_data.to_dict()
                attachments.append(attachments_type_0_item)

        else:
            attachments = self.attachments

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, ConnectorInboundRequestMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        timestamp: None | str | Unset
        if isinstance(self.timestamp, Unset):
            timestamp = UNSET
        else:
            timestamp = self.timestamp

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "event_id": event_id,
                "chat_id": chat_id,
                "content": content,
            }
        )
        if sender is not UNSET:
            field_dict["sender"] = sender
        if attachments is not UNSET:
            field_dict["attachments"] = attachments
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if timestamp is not UNSET:
            field_dict["timestamp"] = timestamp

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connector_inbound_request_attachments_type_0_item import (
            ConnectorInboundRequestAttachmentsType0Item,
        )
        from ..models.connector_inbound_request_metadata_type_0 import (
            ConnectorInboundRequestMetadataType0,
        )
        from ..models.connector_inbound_request_sender import (
            ConnectorInboundRequestSender,
        )

        d = dict(src_dict)
        event_id = d.pop("event_id")

        chat_id = d.pop("chat_id")

        content = d.pop("content")

        _sender = d.pop("sender", UNSET)
        sender: ConnectorInboundRequestSender | Unset
        if isinstance(_sender, Unset):
            sender = UNSET
        else:
            sender = ConnectorInboundRequestSender.from_dict(_sender)

        def _parse_attachments(
            data: object,
        ) -> list[ConnectorInboundRequestAttachmentsType0Item] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                attachments_type_0 = []
                _attachments_type_0 = data
                for attachments_type_0_item_data in _attachments_type_0:
                    attachments_type_0_item = (
                        ConnectorInboundRequestAttachmentsType0Item.from_dict(
                            attachments_type_0_item_data
                        )
                    )

                    attachments_type_0.append(attachments_type_0_item)

                return attachments_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                list[ConnectorInboundRequestAttachmentsType0Item] | None | Unset, data
            )

        attachments = _parse_attachments(d.pop("attachments", UNSET))

        def _parse_metadata(
            data: object,
        ) -> ConnectorInboundRequestMetadataType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = ConnectorInboundRequestMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ConnectorInboundRequestMetadataType0 | None | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        def _parse_timestamp(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        timestamp = _parse_timestamp(d.pop("timestamp", UNSET))

        connector_inbound_request = cls(
            event_id=event_id,
            chat_id=chat_id,
            content=content,
            sender=sender,
            attachments=attachments,
            metadata=metadata,
            timestamp=timestamp,
        )

        return connector_inbound_request
