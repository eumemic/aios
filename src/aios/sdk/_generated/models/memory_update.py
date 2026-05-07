from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.memory_update_precondition import MemoryUpdatePrecondition


T = TypeVar("T", bound="MemoryUpdate")


@_attrs_define
class MemoryUpdate:
    """Request body for ``POST /v1/memory-stores/{store_id}/memories/{id}``.

    Either ``content`` or ``path`` (or both) must be provided. Precondition
    only gates the content half — renames are unconditional, matching the
    Anthropic semantics confirmed by live probe.

        Attributes:
            content (None | str | Unset):
            path (None | str | Unset):
            precondition (MemoryUpdatePrecondition | None | Unset):
    """

    content: None | str | Unset = UNSET
    path: None | str | Unset = UNSET
    precondition: MemoryUpdatePrecondition | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.memory_update_precondition import MemoryUpdatePrecondition

        content: None | str | Unset
        if isinstance(self.content, Unset):
            content = UNSET
        else:
            content = self.content

        path: None | str | Unset
        if isinstance(self.path, Unset):
            path = UNSET
        else:
            path = self.path

        precondition: dict[str, Any] | None | Unset
        if isinstance(self.precondition, Unset):
            precondition = UNSET
        elif isinstance(self.precondition, MemoryUpdatePrecondition):
            precondition = self.precondition.to_dict()
        else:
            precondition = self.precondition

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if content is not UNSET:
            field_dict["content"] = content
        if path is not UNSET:
            field_dict["path"] = path
        if precondition is not UNSET:
            field_dict["precondition"] = precondition

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.memory_update_precondition import MemoryUpdatePrecondition

        d = dict(src_dict)

        def _parse_content(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        content = _parse_content(d.pop("content", UNSET))

        def _parse_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        path = _parse_path(d.pop("path", UNSET))

        def _parse_precondition(
            data: object,
        ) -> MemoryUpdatePrecondition | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                precondition_type_0 = MemoryUpdatePrecondition.from_dict(data)

                return precondition_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(MemoryUpdatePrecondition | None | Unset, data)

        precondition = _parse_precondition(d.pop("precondition", UNSET))

        memory_update = cls(
            content=content,
            path=path,
            precondition=precondition,
        )

        return memory_update
