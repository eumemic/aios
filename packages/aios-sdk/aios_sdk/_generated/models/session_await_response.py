from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_await_response_error_type_0 import (
        SessionAwaitResponseErrorType0,
    )


T = TypeVar("T", bound="SessionAwaitResponse")


@_attrs_define
class SessionAwaitResponse:
    """Response for GET /v1/sessions/{id}/await — the await-a-completion primitive,
    session backing. One envelope over two monotonic predicates (request_id
    correlation, or reacted>=watermark). Poll until `done`.

        Attributes:
            done (bool):
            last_reacted_seq (int):
            result (Any | Unset):
            is_error (bool | Unset):  Default: False.
            error (None | SessionAwaitResponseErrorType0 | Unset):
    """

    done: bool
    last_reacted_seq: int
    result: Any | Unset = UNSET
    is_error: bool | Unset = False
    error: None | SessionAwaitResponseErrorType0 | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.session_await_response_error_type_0 import (
            SessionAwaitResponseErrorType0,
        )

        done = self.done

        last_reacted_seq = self.last_reacted_seq

        result = self.result

        is_error = self.is_error

        error: dict[str, Any] | None | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        elif isinstance(self.error, SessionAwaitResponseErrorType0):
            error = self.error.to_dict()
        else:
            error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "done": done,
                "last_reacted_seq": last_reacted_seq,
            }
        )
        if result is not UNSET:
            field_dict["result"] = result
        if is_error is not UNSET:
            field_dict["is_error"] = is_error
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.session_await_response_error_type_0 import (
            SessionAwaitResponseErrorType0,
        )

        d = dict(src_dict)
        done = d.pop("done")

        last_reacted_seq = d.pop("last_reacted_seq")

        result = d.pop("result", UNSET)

        is_error = d.pop("is_error", UNSET)

        def _parse_error(data: object) -> None | SessionAwaitResponseErrorType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                error_type_0 = SessionAwaitResponseErrorType0.from_dict(data)

                return error_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SessionAwaitResponseErrorType0 | Unset, data)

        error = _parse_error(d.pop("error", UNSET))

        session_await_response = cls(
            done=done,
            last_reacted_seq=last_reacted_seq,
            result=result,
            is_error=is_error,
            error=error,
        )

        session_await_response.additional_properties = d
        return session_await_response

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
