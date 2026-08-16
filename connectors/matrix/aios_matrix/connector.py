"""Supervised Matrix application-service connector."""

from __future__ import annotations

import asyncio
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import structlog
from aiohttp import ClientError, web
from aios_connector_http import HttpConnector, SandboxPath, tool
from aios_connector_http.spool import SqliteAnsweredSpool
from markdown_it import MarkdownIt
from mautrix.appservice import AppService
from mautrix.errors import MatrixConnectionError, MatrixRequestError, MForbidden
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
    """At least one intended recipient has no live connection *yet*.

    Raised instead of returning normally so the transaction is NOT ACKed.
    Matrix appservice delivery is at-least-once: the homeserver retries a
    transaction until it sees a 2xx and drops it permanently once it does.
    Answering 200 for an event we could not route turns a transient
    not-ready condition (``setup()`` starts the listener before discovery
    has populated ``_ghost_connections``) into silent, permanent loss of a
    human's message.  Failing closed makes the homeserver redeliver.

    Raised for PARTIAL routability too, not just total.  "Some ghosts
    present" is not "all ghosts routable": a group room holding one live
    ghost and one not-yet-connected ghost used to emit to the live subset,
    return normally, and ACK 200 — losing the second recipient's copy
    while reporting success.  Any unroutable recipient fails the whole
    transaction.
    """


class RoutingPermanentlyUnroutable(RuntimeError):
    """A recipient has stayed unroutable across so many redeliveries that
    retrying is no longer plausibly productive.

    NOTE — this is an escalation, NOT an ACK.  It still answers non-2xx, so
    the homeserver keeps the message; what changes is that the failure
    becomes LOUD (HALT → container exit → health/dead-man) instead of an
    invisible 503 loop that head-of-line blocks every Matrix transaction
    forever.  Nothing is dropped on this path.

    Why a count and not a classification: from inside the container a
    permanently-deleted connection and a transiently-absent one are not
    reliably distinguishable.  ``_ghost_connections`` is populated by
    ``serve_connection`` and popped in its ``finally``, and the runner
    ALSO pops ``_connections[connection_id]`` on a terminal worker failure
    before re-spawning it with backoff — so "absent" covers both "deleted"
    and "crashed, restarting shortly".  The discovery protocol makes the
    same point normatively: after a ``reset`` the runner treats absence
    from a snapshot as *unknown*, never as *removed*.  Guessing
    "permanent" wrongly would convert a recoverable delay into
    unrecoverable message loss, so permanence is never inferred — only
    persistent failure to make progress is measured.
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
    # How many times one event may be redelivered while a recipient stays
    # unroutable before the condition is escalated from "retry quietly" to
    # "halt loudly".  Synapse's recoverer backs off 2**n seconds capped at
    # ~512 s, so ~12 attempts is on the order of an hour of retrying — long
    # enough to cover container restarts and discovery re-subscribes, short
    # enough that a genuinely dead recipient does not head-of-line block
    # every Matrix transaction indefinitely and unseen.
    MAX_UNROUTABLE_REDELIVERIES = 12
    # Membership repair is on Synapse's ordered transaction path.  Connected
    # ghosts are only candidates (liveness does not prove room membership), so
    # never serially probe the appservice's whole connection population.
    MAX_MEMBERSHIP_REPAIR_CANDIDATES = 8
    # Bounds the redelivery-attempt ledger.  Entries are keyed per event and
    # dropped once the event is fully delivered; this cap only matters if a
    # large number of distinct events are simultaneously stuck.
    _MAX_TRACKED_UNROUTABLE_EVENTS = 4096

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
        # event_id → how many times we have refused it for unroutability.
        # Drives the transient → permanent escalation; see
        # ``RoutingPermanentlyUnroutable``.
        self._unroutable_attempts: dict[str, int] = {}
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
        if isinstance(exc, RoutingPermanentlyUnroutable):
            # Retrying has stopped being productive: this event has been
            # refused ``MAX_UNROUTABLE_REDELIVERIES`` times and a recipient
            # is still unroutable.  HALT does NOT ack and does NOT drop —
            # the homeserver still holds the transaction.  What it does is
            # make an otherwise-invisible infinite 503 loop loud: the
            # container exits, health/dead-man fires, and an operator gets
            # to decide, instead of ordered Matrix traffic silently
            # head-of-line blocking forever behind a stale room member.
            return ReceiverAction.HALT
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

    async def _verified_joined_members(self, event: Event) -> list[UserID]:
        """The room's join list, or ``RoutingNotReady`` if it cannot be trusted.

        ``state_store.get_members()`` is a CACHE, not an authority.  mautrix
        populates it incrementally: ``AppService`` registers
        ``state_store.update_state`` as an event handler, so a member's join
        is learned only when that member's ``m.room.member`` event happens to
        come down a transaction.  The only bulk fill is
        ``get_joined_members()``, which write-throughs via
        ``StoreUpdatingAPI`` and is the one call that sets
        ``has_full_member_list``; the connector makes it at ghost startup and
        then every ``RECONCILE_SECONDS`` (300 s).

        Between those points the store answers with a NON-EMPTY but
        INCOMPLETE list, and nothing in the shape of the answer says so.  A
        partial view is not a smaller truth, it is a DIFFERENT one:

        * a DM whose ghost join has not been seen yet looks like a room with
          no namespaced member — i.e. "genuinely foreign", ACK-and-ignore.
          That is silent, permanent loss of a human's message.
        * a group whose unroutable co-recipient has not been seen yet looks
          fully routable, so the partial-routability refusal never fires and
          that recipient's copy is lost the same way.

        Both defeat the guards above by feeding them an input they never
        validate: they prove "every member I KNOW ABOUT is routable", not
        "every recipient is routable".

        So completeness is CHECKED (``has_full_member_list``), and on a miss
        repaired from the homeserver rather than assumed.  If it can be
        neither confirmed nor repaired we refuse: an unverifiable membership
        view is treated exactly like an empty one, because it carries exactly
        as much authority.  Refusing is safe (Synapse redelivers); ACKing is
        not (Synapse drops the message forever).
        """
        room_id = event.room_id
        store = self.az.state_store
        if await store.has_full_member_list(room_id):
            return list(await store.get_members(room_id, (Membership.JOIN,)))
        # Incomplete: bulk-fill from the homeserver, which is authoritative
        # and also write-throughs into the store for subsequent events. The
        # bot is normally not joined to human↔ghost DMs, so on M_FORBIDDEN try
        # the connected ghost intents: a ghost in the room can perform this
        # read without changing room membership.
        try:
            joined = await self.az.intent.get_joined_members(
                room_id,
                ensure_joined=False,  # type: ignore[call-arg]
            )
        except MForbidden as bot_forbidden:
            joined = None
            last_permanent: Exception = bot_forbidden
            last_transient: Exception | None = None
            candidates = tuple(self._ghost_connections)[: self.MAX_MEMBERSHIP_REPAIR_CANDIDATES]
            for localpart in candidates:
                intent = self.az.intent.user(self._mxid(localpart))
                try:
                    joined = await intent.get_joined_members(
                        room_id,
                        ensure_joined=False,  # type: ignore[call-arg]
                    )
                    break
                except Exception as exc:
                    # A failed candidate says nothing about later candidates.
                    # Keep trying within the strict fan-out bound.
                    if self._is_permanent_membership_failure(exc):
                        last_permanent = exc
                    else:
                        last_transient = exc
            if joined is None:
                if last_transient is not None:
                    # At least one candidate may recover.  Do not spend the
                    # permanent budget merely because the other candidates
                    # cannot read this room.
                    raise RoutingNotReady(
                        f"membership view for room {room_id} is incomplete and bounded ghost "
                        f"repair failed transiently ({type(last_transient).__name__}: "
                        f"{last_transient})"
                    ) from last_transient
                self._raise_unrepairable_membership(event, room_id, last_permanent)
        except Exception as exc:
            if self._is_permanent_membership_failure(exc):
                self._raise_unrepairable_membership(event, room_id, exc)
            # Could not verify — refuse, do NOT ACK.  A transport failure or
            # server-side response may recover and must not consume the
            # permanent budget.
            raise RoutingNotReady(
                f"membership view for room {room_id} is incomplete and could not be "
                f"refreshed ({type(exc).__name__}: {exc}); refusing rather than "
                "classifying on a partial member list"
            ) from exc
        if not joined:
            # The sender is necessarily joined, so an empty authoritative
            # answer is itself untrustworthy.
            raise RoutingNotReady(
                f"homeserver returned no joined members for room {room_id}; cannot classify event"
            )
        return list(joined.keys())

    @staticmethod
    def _is_permanent_membership_failure(exc: Exception) -> bool:
        """Whether a membership read should consume the finite failure budget.

        Retry only failures with an explicit transient contract. Unknown
        exceptions are bounded: treating them as transient would let a code,
        decoding, or unexpected-response failure stall all Matrix traffic
        forever without an operator signal.
        """
        if isinstance(
            exc,
            (
                MatrixConnectionError,
                httpx.TransportError,
                ClientError,
                OSError,
                EOFError,
                TimeoutError,
            ),
        ):
            return False
        if isinstance(exc, json.JSONDecodeError):
            # A truncated 200 can cause this transiently, but malformed JSON is
            # also a stable homeserver/proxy protocol violation. Bound it so a
            # persistently invalid success response cannot stall every room
            # forever; unlike OS/transport errors it has no transient contract.
            return True
        if isinstance(exc, MatrixRequestError):
            status = getattr(exc, "http_status", None)
            # Request timeout and rate limiting are explicitly transient, as
            # are all server failures. Other 4xx answers (notably 404/410) are
            # stable verdicts for this read.
            return not (isinstance(status, int) and (status in (408, 429) or status >= 500))
        return True

    def _raise_unrepairable_membership(
        self, event: Event, room_id: RoomID, cause: Exception
    ) -> None:
        attempts = self._record_unroutable_attempt(event)
        detail = (
            f"membership view for room {room_id} is permanently unrepairable "
            f"({type(cause).__name__}: {cause}, attempt {attempts})"
        )
        if attempts >= self.MAX_UNROUTABLE_REDELIVERIES:
            raise RoutingPermanentlyUnroutable(
                f"{detail}; still unverifiable after "
                f"{self.MAX_UNROUTABLE_REDELIVERIES} redeliveries"
            ) from cause
        raise RoutingNotReady(detail) from cause

    async def _handle_event(self, event: Event) -> None:
        if event.type != EventType.ROOM_MESSAGE or not getattr(event.content, "body", None):
            return
        # VERIFIED membership, not merely cached membership.  Everything
        # below — who the receivers are, whether they are all routable, and
        # whether the room is a dm — is only as sound as this list.
        members = await self._verified_joined_members(event)
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
        # PARTIAL routability is the same failure as total unroutability.
        # The pre-fix code asked "are there ANY live ghosts?" and, if so,
        # emitted to that subset and returned normally — ACKing 200 while a
        # co-recipient's copy was never delivered.  Synapse drops a
        # transaction permanently once it sees a 2xx, so that ACK is silent,
        # unrecoverable loss of a human's message.  The question must be
        # "are ALL intended recipients routable?".
        unroutable = [
            member for member in receivers if self._localpart(member) not in self._ghost_connections
        ]
        if unroutable:
            # Deliver NOTHING on a partial-routability failure.  Emitting to
            # the live subset first and then refusing would double-deliver
            # to that subset on every redelivery *if* the emit were not
            # idempotent; it is idempotent (see below), but emitting before
            # a refusal still buys nothing and costs an extra aios round
            # trip per retry, so the whole event is failed atomically.
            attempts = self._record_unroutable_attempt(event)
            log.warning(
                "matrix.inbound.routing_not_ready",
                room_id=str(event.room_id),
                event_id=str(event.event_id),
                namespaced_members=[str(member) for member in receivers],
                unroutable_members=[str(member) for member in unroutable],
                live_members=[str(member) for member in receivers if member not in set(unroutable)],
                attempts=attempts,
            )
            detail = (
                f"no live connection for {[str(member) for member in unroutable]} "
                f"in room {event.room_id} "
                f"(recipients={[str(member) for member in receivers]}, attempt {attempts})"
            )
            if attempts >= self.MAX_UNROUTABLE_REDELIVERIES:
                # Escalate LOUD, do not ACK.  See RoutingPermanentlyUnroutable.
                raise RoutingPermanentlyUnroutable(
                    f"{detail}; still unroutable after "
                    f"{self.MAX_UNROUTABLE_REDELIVERIES} redeliveries"
                )
            raise RoutingNotReady(detail)
        room_kind = "dm" if len(members) == 2 else "group"
        # Every recipient is routable.  The emits below are individually
        # idempotent — ``event_id`` is the deterministic
        # ``matrix-{localpart}-{matrix event id}``, and the aios inbound
        # endpoint dedups on (account, connector, external_account_id,
        # event_id) with an ON CONFLICT DO NOTHING ledger row written in the
        # same transaction as the append.  So if a LATER event in the same
        # Synapse transaction fails, redelivery of this whole transaction
        # re-emits these copies harmlessly rather than duplicating them.
        for ghost in receivers:
            localpart = self._localpart(ghost)
            await self.emit_inbound(
                connection_id=self._ghost_connections[localpart],
                chat_id=str(event.room_id),
                sender={"display_name": str(event.sender), "mxid": str(event.sender)},
                content=event.content.body,
                event_id=f"matrix-{localpart}-{event.event_id}",
                metadata={"room_kind": room_kind},
            )
        # Fully delivered: stop tracking redelivery attempts for this event.
        self._unroutable_attempts.pop(str(event.event_id), None)

    def _record_unroutable_attempt(self, event: Event) -> int:
        """Count how many times this event has been refused as unroutable.

        Keyed on the Matrix event id, which is stable across redeliveries of
        the same transaction, so the count measures "how long has this
        specific message been stuck", not "how busy is the server".

        The counter is per-process: a container restart resets it.  That is
        deliberate and is the safe direction — a restart is exactly the
        event most likely to FIX routability, so it earns a fresh budget
        rather than inheriting a stale one and halting immediately.
        """
        key = str(event.event_id)
        if key not in self._unroutable_attempts and (
            len(self._unroutable_attempts) >= self._MAX_TRACKED_UNROUTABLE_EVENTS
        ):
            # Bounded ledger: drop the oldest tracked event (dicts preserve
            # insertion order).  Losing a count only costs that event a
            # fresh retry budget — it never causes an ACK or a drop.
            self._unroutable_attempts.pop(next(iter(self._unroutable_attempts)), None)
        attempts = self._unroutable_attempts.get(key, 0) + 1
        self._unroutable_attempts[key] = attempts
        return attempts

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
