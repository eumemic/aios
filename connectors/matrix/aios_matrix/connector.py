"""Supervised Matrix application-service connector."""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import structlog
from aiohttp import web
from aios_connector_http import HttpConnector, SandboxPath, tool
from aios_connector_http.spool import SqliteAnsweredSpool
from markdown_it import MarkdownIt
from mautrix.appservice import AppService
from mautrix.types import (
    Event,
    EventID,
    EventType,
    Format,
    InReplyTo,
    Membership,
    MessageType,
    RelatesTo,
    RoomID,
    TextMessageEventContent,
    UserID,
)

from .appservice import create_appservice
from .config import MatrixConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class ReceiverAction(Enum):
    RETRY = "retry"
    HALT = "halt"


class ReceiverHalted(RuntimeError):
    pass


class RoutingNotReady(RuntimeError):
    """A namespaced ghost is addressable but has no live connection yet.

    Raised instead of returning normally so the transaction is NOT ACKed.
    Matrix appservice delivery is at-least-once: the homeserver retries a
    transaction until it sees a 2xx and drops it permanently once it does.
    Answering 200 for an event we could not route turns a transient
    not-ready condition (``setup()`` starts the listener before discovery
    has populated ``_ghost_connections``) into silent, permanent loss of a
    human's message.  Failing closed makes the homeserver redeliver.
    """


class SupervisedAppService(AppService):
    """mautrix receiver that does not turn every handler failure into a 200."""

    connector: MatrixConnector

    async def _http_handle_transaction(self, request: web.Request) -> web.Response:
        try:
            transaction_id, data = await self._read_transaction_header(request)
            events = data.pop("events")
            await self.handle_transaction(transaction_id, events=events, extra_data=cast(Any, data))
        except web.HTTPException:
            raise
        except Exception as exc:
            action = await self.connector._receiver_failed(exc)
            status = 503 if action is ReceiverAction.RETRY else 500
            return web.json_response({"error": action.value}, status=status)
        self.transactions.add(transaction_id)
        return web.json_response({})


class MatrixConnector(HttpConnector):
    connector = "matrix"
    uses_connection_secrets = False
    RECONCILE_SECONDS = 300.0

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        spool_path: str | Path | None = None,
    ) -> None:
        super().__init__(base_url=base_url, token=token)
        self.config: MatrixConfig | None = None
        self.az: SupervisedAppService | Any = None
        self._halt = asyncio.Event()
        self._halt_error: BaseException | None = None
        self._ghost_connections: dict[str, str] = {}
        self._spool = SqliteAnsweredSpool(
            spool_path or Path("/var/lib/aios-matrix/answered.sqlite")
        )

    async def load_answered(self) -> dict[str, str | None]:
        return self._spool.load()

    async def save_answered(self, tool_call_id: str, result: str | None = None) -> None:
        self._spool.add(tool_call_id, result)

    async def setup(self, tg: asyncio.TaskGroup) -> None:
        self.config = MatrixConfig()
        self.az = cast(
            SupervisedAppService,
            create_appservice(self.config, appservice_class=SupervisedAppService),
        )
        self.az.connector = self
        self.az.synchronous_handlers = True
        self.az.matrix_event_handler(self._handle_event)
        host, port = self.config.listen
        await self.az.start(host, port)
        tg.create_task(self._halt_supervisor())
        tg.create_task(self._periodic_reconcile())

    async def teardown(self) -> None:
        if self.az is not None:
            await self.az.stop()
        self._spool.close()

    async def _halt_supervisor(self) -> None:
        await self._halt.wait()
        raise ReceiverHalted("Matrix receiver halted") from self._halt_error

    def _classify_receiver_failure(self, exc: BaseException) -> ReceiverAction:
        if isinstance(exc, RoutingNotReady):
            # Transient by construction: the connection worker that
            # populates ``_ghost_connections`` is still starting.  Must NOT
            # fall through to the HALT default — a routine startup race
            # would otherwise tear the whole container down.
            return ReceiverAction.RETRY
        if isinstance(exc, httpx.TransportError):
            return ReceiverAction.RETRY
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status >= 500:
                return ReceiverAction.RETRY
            if status in (401, 403):
                return ReceiverAction.HALT
        return ReceiverAction.HALT

    async def _receiver_failed(self, exc: BaseException) -> ReceiverAction:
        action = self._classify_receiver_failure(exc)
        log.error("matrix.receiver.failed", action=action.value, error=str(exc))
        if action is ReceiverAction.HALT:
            self._halt_error = exc
            self._halt.set()
        return action

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        del secrets
        state = self._connections[connection_id]
        localpart = state.external_account_id
        self._ghost_connections[localpart] = connection_id
        intent = self.az.intent.user(self._mxid(localpart))
        await intent.ensure_registered()
        await self._reconcile_intent(intent)
        try:
            await asyncio.Event().wait()
        finally:
            self._ghost_connections.pop(localpart, None)

    async def _periodic_reconcile(self) -> None:
        while True:
            await asyncio.sleep(self.RECONCILE_SECONDS)
            for localpart in tuple(self._ghost_connections):
                await self._reconcile_intent(self.az.intent.user(self._mxid(localpart)))

    async def _reconcile_intent(self, intent: Any) -> None:
        for room_id in await intent.get_joined_rooms():
            await intent.get_joined_members(room_id)

    def _mxid(self, localpart: str) -> UserID:
        assert self.config is not None
        return UserID(f"@{localpart}:{self.config.server_name}")

    def _is_ghost(self, user_id: UserID) -> bool:
        assert self.config is not None
        return re.fullmatch(self.config.user_namespace_regex, str(user_id)) is not None

    @staticmethod
    def _localpart(user_id: UserID) -> str:
        return str(user_id).removeprefix("@").split(":", 1)[0]

    async def _handle_event(self, event: Event) -> None:
        if event.type != EventType.ROOM_MESSAGE or not getattr(event.content, "body", None):
            return
        members = await self.az.state_store.get_members(event.room_id, (Membership.JOIN,))
        # Three distinct facts, which must NOT be collapsed into one
        # "nothing to do" branch — the conflation is what loses messages:
        #
        #   1. no membership state    → we cannot classify → retry
        #   2. no namespaced receiver → genuinely not ours → ACK + ignore
        #   3. receiver but no route  → not ready yet      → retry
        #
        # A room always contains at least the event's own sender, so an
        # empty member list never means "an empty room"; it means the state
        # store has nothing for this room (cold store, un-reconciled ghost).
        # Treating that as "not ours" would silently drop a real DM.
        if not members:
            raise RoutingNotReady(
                f"no membership state for room {event.room_id}; cannot classify event"
            )
        # Namespaced members other than the sender are the candidate
        # receivers.  Excluding the sender here (rather than after the
        # routing check) keeps our own ghost's echo in the ACK-and-ignore
        # case: it is not undeliverable, it is simply not inbound traffic.
        receivers = [
            member for member in members if self._is_ghost(member) and member != event.sender
        ]
        if not receivers:
            return
        ghosts = [
            member for member in receivers if self._localpart(member) in self._ghost_connections
        ]
        if not ghosts:
            log.warning(
                "matrix.inbound.routing_not_ready",
                room_id=str(event.room_id),
                namespaced_members=[str(member) for member in receivers],
            )
            raise RoutingNotReady(
                f"no live connection for {[str(member) for member in receivers]} "
                f"in room {event.room_id}"
            )
        room_kind = "dm" if len(members) == 2 else "group"
        for ghost in ghosts:
            localpart = self._localpart(ghost)
            await self.emit_inbound(
                connection_id=self._ghost_connections[localpart],
                chat_id=str(event.room_id),
                sender={"display_name": str(event.sender), "mxid": str(event.sender)},
                content=event.content.body,
                event_id=f"matrix-{localpart}-{event.event_id}",
                metadata={"room_kind": room_kind},
            )

    @tool(name="matrix_send", delivery=True)
    async def matrix_send(
        self,
        *,
        connection_id: str,
        external_account_id: str,
        chat_id: str,
        text: str,
        format: Literal["plain", "markdown"] = "plain",
        reply_to: str | None = None,
        attachments: list[SandboxPath] | None = None,
        tool_call_id: str,
    ) -> dict[str, str]:
        """Send a message to the current Matrix room."""
        del connection_id
        intent = self.az.intent.user(self._mxid(external_account_id))
        content = TextMessageEventContent(msgtype=MessageType.TEXT, body=text)
        if format == "markdown":
            content.format = Format.HTML
            content.formatted_body = MarkdownIt("commonmark", {"html": False}).render(text)
        if reply_to:
            content.relates_to = RelatesTo(in_reply_to=InReplyTo(event_id=EventID(reply_to)))
        if attachments:
            uploaded = []
            for attachment in attachments:
                data = attachment.read_bytes()
                uploaded.append(
                    str(await intent.upload_media(data, filename=attachment.name, size=len(data)))
                )
            content["com.aios.attachments"] = uploaded
        event_id = await intent.send_message_event(
            RoomID(chat_id), EventType.ROOM_MESSAGE, content, txn_id=tool_call_id
        )
        return {"event_id": str(event_id), "room_id": chat_id}
