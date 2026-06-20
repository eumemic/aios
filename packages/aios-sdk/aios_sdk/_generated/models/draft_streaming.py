from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="DraftStreaming")


@_attrs_define
class DraftStreaming:
    """Present == connector can render in-progress assistant deltas as an
    editable draft.  Absent == connector waits for the committed message.

        Attributes:
            overflow_limit (int | None | Unset):
    """

    overflow_limit: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        overflow_limit: int | None | Unset
        if isinstance(self.overflow_limit, Unset):
            overflow_limit = UNSET
        else:
            overflow_limit = self.overflow_limit

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if overflow_limit is not UNSET:
            field_dict["overflow_limit"] = overflow_limit

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_overflow_limit(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        overflow_limit = _parse_overflow_limit(d.pop("overflow_limit", UNSET))

        draft_streaming = cls(
            overflow_limit=overflow_limit,
        )

        return draft_streaming
