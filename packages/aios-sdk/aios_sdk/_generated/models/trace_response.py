from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.trace_response_root_kind import TraceResponseRootKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.trace_entry import TraceEntry
    from ..models.trace_truncated import TraceTruncated


T = TypeVar("T", bound="TraceResponse")


@_attrs_define
class TraceResponse:
    """A one-call flat trace for a run-root or session-root.

    ``root_kind`` / ``root_id`` name the tree's root. ``entries`` is the flat
    DFS-pre-order list (nodes interleaved with their journal frames).
    ``truncated`` is non-``None`` iff the node-count ceiling cut the walk.

    Ordering caveat (documented per #1149): cross-subtree time-ordering is
    best-effort to **transaction granularity** — ``created_at`` is
    ``transaction_timestamp()`` (constant for a whole run-step's journal; it can
    tie or invert across concurrent transactions on separate pooled
    connections), and the two journals share no global sequence. Only the causal
    parent→child edge is exact, so DFS pre-order is canonical; chronological is a
    client-side re-sort.

    Scope caveat: ``wake_session`` peer-pokes are out of scope (a peer stimulus,
    not a spawn — no ``request_opened`` edge); they do not appear as nodes.

        Attributes:
            root_kind (TraceResponseRootKind):
            root_id (str):
            entries (list[TraceEntry] | Unset):
            truncated (None | TraceTruncated | Unset):
    """

    root_kind: TraceResponseRootKind
    root_id: str
    entries: list[TraceEntry] | Unset = UNSET
    truncated: None | TraceTruncated | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.trace_truncated import TraceTruncated

        root_kind = self.root_kind.value

        root_id = self.root_id

        entries: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.entries, Unset):
            entries = []
            for entries_item_data in self.entries:
                entries_item = entries_item_data.to_dict()
                entries.append(entries_item)

        truncated: dict[str, Any] | None | Unset
        if isinstance(self.truncated, Unset):
            truncated = UNSET
        elif isinstance(self.truncated, TraceTruncated):
            truncated = self.truncated.to_dict()
        else:
            truncated = self.truncated

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "root_kind": root_kind,
                "root_id": root_id,
            }
        )
        if entries is not UNSET:
            field_dict["entries"] = entries
        if truncated is not UNSET:
            field_dict["truncated"] = truncated

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trace_entry import TraceEntry
        from ..models.trace_truncated import TraceTruncated

        d = dict(src_dict)
        root_kind = TraceResponseRootKind(d.pop("root_kind"))

        root_id = d.pop("root_id")

        _entries = d.pop("entries", UNSET)
        entries: list[TraceEntry] | Unset = UNSET
        if _entries is not UNSET:
            entries = []
            for entries_item_data in _entries:
                entries_item = TraceEntry.from_dict(entries_item_data)

                entries.append(entries_item)

        def _parse_truncated(data: object) -> None | TraceTruncated | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                truncated_type_0 = TraceTruncated.from_dict(data)

                return truncated_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TraceTruncated | Unset, data)

        truncated = _parse_truncated(d.pop("truncated", UNSET))

        trace_response = cls(
            root_kind=root_kind,
            root_id=root_id,
            entries=entries,
            truncated=truncated,
        )

        trace_response.additional_properties = d
        return trace_response

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
