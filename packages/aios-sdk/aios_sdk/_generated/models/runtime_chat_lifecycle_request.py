from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.runtime_chat_lifecycle_request_data_type_0 import (
        RuntimeChatLifecycleRequestDataType0,
    )


T = TypeVar("T", bound="RuntimeChatLifecycleRequest")


@_attrs_define
class RuntimeChatLifecycleRequest:
    """Body for ``POST /v1/connectors/runtime/chat-lifecycle`` (#1260).

    The routing-key variant of :class:`RuntimeSessionLifecycleRequest`.
    Both target a *single* session (not the broadcast fan-out), but where
    the session-lifecycle route needs the caller to already hold the
    resolved ``session_id``, this route carries a per-peer **routing key**
    (``chat_id``) and resolves it through the connection's per-chat binding
    to the originating session server-side.

    This is the second option the SMS design (§3.5 req 1) calls out: "route
    the per-peer failure through the resolver on the callback's ``To``".  A
    Twilio status callback knows the peer number (→ ``chat_id``) but not the
    AIOS ``session_id`` — without this route the connector would have to do
    an extra round-trip (or maintain its own ``chat_id → session_id`` map)
    just to reach the originating per_chat session.  The broadcast
    ``/runtime/lifecycle`` route stays for genuine connection-wide events.

    ``chat_id`` is the connector's per-peer routing key, the same value the
    inbound path stamps onto ``chat_sessions``.  It must resolve to an
    existing per-chat binding on ``connection_id`` — a routing key with no
    bound session 404s rather than fanning a spurious cross-peer notice (the
    design's "if a correlation row is genuinely missing … drop rather than
    fan a spurious cross-peer failure", §3.5).

    ``wake`` mirrors the session-lifecycle route: ``True`` pairs the append
    with a ``defer_wake`` so the failure wakes the originating session;
    defaults ``False`` (visible-on-next-turn).

        Attributes:
            connection_id (str):
            chat_id (str):
            event (str):
            reason (None | str | Unset):
            data (None | RuntimeChatLifecycleRequestDataType0 | Unset):
            wake (bool | Unset):  Default: False.
    """

    connection_id: str
    chat_id: str
    event: str
    reason: None | str | Unset = UNSET
    data: None | RuntimeChatLifecycleRequestDataType0 | Unset = UNSET
    wake: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        from ..models.runtime_chat_lifecycle_request_data_type_0 import (
            RuntimeChatLifecycleRequestDataType0,
        )

        connection_id = self.connection_id

        chat_id = self.chat_id

        event = self.event

        reason: None | str | Unset
        if isinstance(self.reason, Unset):
            reason = UNSET
        else:
            reason = self.reason

        data: dict[str, Any] | None | Unset
        if isinstance(self.data, Unset):
            data = UNSET
        elif isinstance(self.data, RuntimeChatLifecycleRequestDataType0):
            data = self.data.to_dict()
        else:
            data = self.data

        wake = self.wake

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connection_id": connection_id,
                "chat_id": chat_id,
                "event": event,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason
        if data is not UNSET:
            field_dict["data"] = data
        if wake is not UNSET:
            field_dict["wake"] = wake

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.runtime_chat_lifecycle_request_data_type_0 import (
            RuntimeChatLifecycleRequestDataType0,
        )

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        chat_id = d.pop("chat_id")

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
        ) -> None | RuntimeChatLifecycleRequestDataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_type_0 = RuntimeChatLifecycleRequestDataType0.from_dict(data)

                return data_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | RuntimeChatLifecycleRequestDataType0 | Unset, data)

        data = _parse_data(d.pop("data", UNSET))

        wake = d.pop("wake", UNSET)

        runtime_chat_lifecycle_request = cls(
            connection_id=connection_id,
            chat_id=chat_id,
            event=event,
            reason=reason,
            data=data,
            wake=wake,
        )

        return runtime_chat_lifecycle_request
