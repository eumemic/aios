from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.obligation_output_schema_type_0 import ObligationOutputSchemaType0


T = TypeVar("T", bound="Obligation")


@_attrs_define
class Obligation:
    """One still-open **awaited** request the session owes a response to (#1413).

    The dual of :class:`AwaitingToolCall`: that view lists tool calls the session
    is *blocked on*; this one lists requests the session *owes an answer to*. An
    obligation is an open ``request_opened`` edge (#1123) — ``awaited=true``, with
    no paired ``request_response`` — and is the in-context surface the model reads
    to know which ``request_id`` to echo back to ``return``/``error``.

    Derived (oldest-first) from the trusted ``request_opened`` lifecycle frame via
    :func:`aios.db.queries.sessions.get_open_obligations`, NEVER the forgeable
    ``metadata.request`` user-message blob (#1131-proof). ``caller_kind`` is the
    trusted ``caller.kind`` (``api``|``session``|``run``); ``opened_at`` is the
    edge's ``created_at`` (for age); ``summary`` is a short truncated preview of
    the request input (absent on pre-#1413 frames → ``None``, rendered id-only).

    ``output_schema`` (#1522) is the JSON Schema the request demands of its
    response ``value`` — the **acceptance contract** the session must produce to
    answer. It is the same datum :func:`aios.db.queries.sessions.get_request_output_schema`
    reads off the ``request_opened`` frame, now projected directly onto the owed
    read-model so a single renderer can show "here is what you owe **and the
    format**". Additive: ``None`` when the request demands no schema (the common
    case) or on a pre-#1522 frame — no migration.

        Attributes:
            request_id (str):
            caller_kind (str):
            opened_at (datetime.datetime):
            caller_id (None | str | Unset):
            summary (None | str | Unset):
            output_schema (None | ObligationOutputSchemaType0 | Unset):
    """

    request_id: str
    caller_kind: str
    opened_at: datetime.datetime
    caller_id: None | str | Unset = UNSET
    summary: None | str | Unset = UNSET
    output_schema: None | ObligationOutputSchemaType0 | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.obligation_output_schema_type_0 import ObligationOutputSchemaType0

        request_id = self.request_id

        caller_kind = self.caller_kind

        opened_at = self.opened_at.isoformat()

        caller_id: None | str | Unset
        if isinstance(self.caller_id, Unset):
            caller_id = UNSET
        else:
            caller_id = self.caller_id

        summary: None | str | Unset
        if isinstance(self.summary, Unset):
            summary = UNSET
        else:
            summary = self.summary

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, ObligationOutputSchemaType0):
            output_schema = self.output_schema.to_dict()
        else:
            output_schema = self.output_schema

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_id": request_id,
                "caller_kind": caller_kind,
                "opened_at": opened_at,
            }
        )
        if caller_id is not UNSET:
            field_dict["caller_id"] = caller_id
        if summary is not UNSET:
            field_dict["summary"] = summary
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.obligation_output_schema_type_0 import ObligationOutputSchemaType0

        d = dict(src_dict)
        request_id = d.pop("request_id")

        caller_kind = d.pop("caller_kind")

        opened_at = isoparse(d.pop("opened_at"))

        def _parse_caller_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        caller_id = _parse_caller_id(d.pop("caller_id", UNSET))

        def _parse_summary(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        summary = _parse_summary(d.pop("summary", UNSET))

        def _parse_output_schema(
            data: object,
        ) -> None | ObligationOutputSchemaType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = ObligationOutputSchemaType0.from_dict(data)

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ObligationOutputSchemaType0 | Unset, data)

        output_schema = _parse_output_schema(d.pop("output_schema", UNSET))

        obligation = cls(
            request_id=request_id,
            caller_kind=caller_kind,
            opened_at=opened_at,
            caller_id=caller_id,
            summary=summary,
            output_schema=output_schema,
        )

        obligation.additional_properties = d
        return obligation

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
