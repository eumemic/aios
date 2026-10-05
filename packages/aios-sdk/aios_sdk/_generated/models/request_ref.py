from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="RequestRef")


@_attrs_define
class RequestRef:
    """A reference to a request a session sent: the span that captured it (#2471),
    named the way ``GET /v1/sessions/{session_id}/requests/{request_id}`` names it.

    A ref means something only in a field typed as one. A run resolves only the ref
    it was created with (``WfRun.request_ref``), never a ref-shaped value it finds
    in its input or in a tool result.

        Attributes:
            session_id (str):
            request_id (str):
    """

    session_id: str
    request_id: str

    def to_dict(self) -> dict[str, Any]:
        session_id = self.session_id

        request_id = self.request_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "session_id": session_id,
                "request_id": request_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        session_id = d.pop("session_id")

        request_id = d.pop("request_id")

        request_ref = cls(
            session_id=session_id,
            request_id=request_id,
        )

        return request_ref
