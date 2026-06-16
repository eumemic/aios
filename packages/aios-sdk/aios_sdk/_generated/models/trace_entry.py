from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.trace_entry_kind import TraceEntryKind
from ..models.trace_entry_terminal_state_type_0 import TraceEntryTerminalStateType0
from ..types import UNSET, Unset

T = TypeVar("T", bound="TraceEntry")


@_attrs_define
class TraceEntry:
    """One line of a flattened trace.

    ``depth`` is the node's DFS depth (0 = root); a CLI renders it as
    indentation. ``parent_id`` is the enclosing node's id (``None`` for the
    root). For **node** entries ``terminal_state`` / ``error_kind`` carry the
    normalized outcome; for **frame** entries they are ``None`` and ``summary``
    carries the per-kind digest. ``timestamp`` is the entry's ``created_at``
    (``transaction_timestamp()`` for journal frames) — exposed as a column for
    chronology, NOT the canonical order (see the module docstring).

        Attributes:
            depth (int):
            kind (TraceEntryKind):
            id (str):
            timestamp (datetime.datetime | None | Unset):
            parent_id (None | str | Unset):
            summary (str | Unset):  Default: ''.
            terminal_state (None | TraceEntryTerminalStateType0 | Unset):
            error_kind (None | str | Unset):
    """

    depth: int
    kind: TraceEntryKind
    id: str
    timestamp: datetime.datetime | None | Unset = UNSET
    parent_id: None | str | Unset = UNSET
    summary: str | Unset = ""
    terminal_state: None | TraceEntryTerminalStateType0 | Unset = UNSET
    error_kind: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        depth = self.depth

        kind = self.kind.value

        id = self.id

        timestamp: None | str | Unset
        if isinstance(self.timestamp, Unset):
            timestamp = UNSET
        elif isinstance(self.timestamp, datetime.datetime):
            timestamp = self.timestamp.isoformat()
        else:
            timestamp = self.timestamp

        parent_id: None | str | Unset
        if isinstance(self.parent_id, Unset):
            parent_id = UNSET
        else:
            parent_id = self.parent_id

        summary = self.summary

        terminal_state: None | str | Unset
        if isinstance(self.terminal_state, Unset):
            terminal_state = UNSET
        elif isinstance(self.terminal_state, TraceEntryTerminalStateType0):
            terminal_state = self.terminal_state.value
        else:
            terminal_state = self.terminal_state

        error_kind: None | str | Unset
        if isinstance(self.error_kind, Unset):
            error_kind = UNSET
        else:
            error_kind = self.error_kind

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "depth": depth,
                "kind": kind,
                "id": id,
            }
        )
        if timestamp is not UNSET:
            field_dict["timestamp"] = timestamp
        if parent_id is not UNSET:
            field_dict["parent_id"] = parent_id
        if summary is not UNSET:
            field_dict["summary"] = summary
        if terminal_state is not UNSET:
            field_dict["terminal_state"] = terminal_state
        if error_kind is not UNSET:
            field_dict["error_kind"] = error_kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        depth = d.pop("depth")

        kind = TraceEntryKind(d.pop("kind"))

        id = d.pop("id")

        def _parse_timestamp(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                timestamp_type_0 = isoparse(data)

                return timestamp_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        timestamp = _parse_timestamp(d.pop("timestamp", UNSET))

        def _parse_parent_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        parent_id = _parse_parent_id(d.pop("parent_id", UNSET))

        summary = d.pop("summary", UNSET)

        def _parse_terminal_state(
            data: object,
        ) -> None | TraceEntryTerminalStateType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                terminal_state_type_0 = TraceEntryTerminalStateType0(data)

                return terminal_state_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TraceEntryTerminalStateType0 | Unset, data)

        terminal_state = _parse_terminal_state(d.pop("terminal_state", UNSET))

        def _parse_error_kind(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        error_kind = _parse_error_kind(d.pop("error_kind", UNSET))

        trace_entry = cls(
            depth=depth,
            kind=kind,
            id=id,
            timestamp=timestamp,
            parent_id=parent_id,
            summary=summary,
            terminal_state=terminal_state,
            error_kind=error_kind,
        )

        trace_entry.additional_properties = d
        return trace_entry

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
