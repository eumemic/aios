from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_user_message_metadata import SessionUserMessageMetadata


T = TypeVar("T", bound="SessionUserMessage")


@_attrs_define
class SessionUserMessage:
    """Request body for `POST /v1/sessions/{id}/messages`.

    Attributes:
        content (str):
        metadata (SessionUserMessageMetadata | Unset):
    """

    content: str
    metadata: SessionUserMessageMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        content = self.content

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "content": content,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.session_user_message_metadata import SessionUserMessageMetadata

        d = dict(src_dict)
        content = d.pop("content")

        _metadata = d.pop("metadata", UNSET)
        metadata: SessionUserMessageMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = SessionUserMessageMetadata.from_dict(_metadata)

        session_user_message = cls(
            content=content,
            metadata=metadata,
        )

        return session_user_message
