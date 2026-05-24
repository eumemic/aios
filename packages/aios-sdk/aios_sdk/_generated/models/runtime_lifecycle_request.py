from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.runtime_lifecycle_request_data_type_0 import (
        RuntimeLifecycleRequestDataType0,
    )


T = TypeVar("T", bound="RuntimeLifecycleRequest")


@_attrs_define
class RuntimeLifecycleRequest:
    """Body for ``POST /v1/connectors/runtime/lifecycle``.

    Lets a connector emit a lifecycle event onto each session bound to
    ``connection_id`` — used today for "the underlying transport just
    went away" notifications (WhatsApp daemon crashed, peer logged the
    device out, etc.) so the model sees the connection-broken state in
    its context instead of silently failing the next outbound.

    ``event`` is a connector-namespaced kind ("whatsapp.connection.lost",
    "signal.daemon.exited") — the connector chooses the vocabulary.
    ``reason`` is an optional short tag the harness surfaces alongside
    the event for the model to act on ("daemon_crashed", "peer_logout").
    ``data`` is an optional free-form dict for connector-specific
    context (current device count, last successful timestamp, etc.).

        Attributes:
            connection_id (str):
            event (str):
            reason (None | str | Unset):
            data (None | RuntimeLifecycleRequestDataType0 | Unset):
    """

    connection_id: str
    event: str
    reason: None | str | Unset = UNSET
    data: None | RuntimeLifecycleRequestDataType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.runtime_lifecycle_request_data_type_0 import (
            RuntimeLifecycleRequestDataType0,
        )

        connection_id = self.connection_id

        event = self.event

        reason: None | str | Unset
        if isinstance(self.reason, Unset):
            reason = UNSET
        else:
            reason = self.reason

        data: dict[str, Any] | None | Unset
        if isinstance(self.data, Unset):
            data = UNSET
        elif isinstance(self.data, RuntimeLifecycleRequestDataType0):
            data = self.data.to_dict()
        else:
            data = self.data

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connection_id": connection_id,
                "event": event,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason
        if data is not UNSET:
            field_dict["data"] = data

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.runtime_lifecycle_request_data_type_0 import (
            RuntimeLifecycleRequestDataType0,
        )

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        event = d.pop("event")

        def _parse_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        reason = _parse_reason(d.pop("reason", UNSET))

        def _parse_data(
            data: object,
        ) -> None | RuntimeLifecycleRequestDataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_type_0 = RuntimeLifecycleRequestDataType0.from_dict(data)

                return data_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | RuntimeLifecycleRequestDataType0 | Unset, data)

        data = _parse_data(d.pop("data", UNSET))

        runtime_lifecycle_request = cls(
            connection_id=connection_id,
            event=event,
            reason=reason,
            data=data,
        )

        return runtime_lifecycle_request
