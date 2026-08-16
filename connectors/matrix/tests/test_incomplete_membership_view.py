"""An INCOMPLETE membership view must be treated exactly like an EMPTY one.

The receiver's whole delivery-safety argument rests on the room's join list:
who the namespaced receivers are, whether every one of them is routable, and
whether the room is a ``dm`` are all read off it.  That list comes from
``state_store.get_members()``, which is a **cache, not an authority**.

mautrix fills it two ways:

* **incrementally** — ``AppService`` registers ``state_store.update_state`` as
  an event handler, so a member's join is learned only if that member's
  ``m.room.member`` event happens to come down a transaction; and
* **in bulk** — ``get_joined_members()`` write-throughs via
  ``StoreUpdatingAPI`` and is the *only* call that sets
  ``has_full_member_list``.  The connector makes it at ghost startup and then
  every ``RECONCILE_SECONDS`` (300 s).

Between those points the store returns a **non-empty but incomplete** list and
**nothing in the shape of the answer says so**.  A partial view is not a
smaller truth, it is a different one, and it defeats the existing guards by
feeding them an input they never validate — they prove *"every member I know
about is routable"*, not *"every recipient is routable"*:

* a DM whose ghost join has not been seen yet looks like a room with **no
  namespaced member**, i.e. "genuinely foreign" → ACK-and-ignore → the human's
  message is gone, with no retry and no log of a refusal;
* a group whose unroutable co-recipient has not been seen yet looks **fully
  routable**, so the partial-routability refusal never fires;
* a partly-known group looks like a **dm** (``len(members) == 2``).

Note the existing ``if not members: raise`` guard is justified in-code by "a
room always contains at least the event's own sender" — which is correct, and
is exactly why it is insufficient: the same argument makes a list containing
*only* the sender equally uninformative, yet ``len == 1`` sailed through as
authoritative.

These tests drive the real aiohttp transaction endpoint, because the defect is
the status code the HTTP layer returns.
"""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from mautrix.appservice.state_store.memory import ASStateStore
from mautrix.client.state_store.memory import MemoryStateStore
from mautrix.errors import MatrixResponseError, MForbidden, MNotFound, MUnknown
from mautrix.types import Member, Membership, RoomID, UserID

from aios_matrix.appservice import create_appservice
from aios_matrix.config import MatrixConfig
from aios_matrix.connector import MatrixConnector, SupervisedAppService

HS_TOKEN = "hs-secret"
ROOM = RoomID("!dm:your.server")
GHOST = UserID("@_aios_agent_one:your.server")
GHOST2 = UserID("@_aios_agent_two:your.server")
HUMAN = UserID("@human:your.server")
HUMAN2 = UserID("@human2:your.server")


class MemoryASStateStore(MemoryStateStore, ASStateStore):
    def __init__(self) -> None:
        MemoryStateStore.__init__(self)
        ASStateStore.__init__(self)


@pytest.fixture
def config() -> MatrixConfig:
    return MatrixConfig(
        hs_url="http://synapse:8008",
        server_name="your.server",
        as_token="as-secret",
        hs_token=HS_TOKEN,
        sender_localpart="_aios",
        user_namespace_regex=r"^@_aios_agent_[a-z0-9]+:your\.server$",
        listen_addr="127.0.0.1:29328",
        database_url="postgresql://unused",
    )


@pytest.fixture
async def receiver(config: MatrixConfig, tmp_path):
    """A live appservice whose state store has learned membership INCREMENTALLY.

    The rooms here are built with ``set_membership`` (one member event at a
    time, as a timeline event would arrive) and therefore are NOT marked
    ``has_full_member_list`` -- which is precisely the state the 300 s
    reconcile window leaves the store in.
    """
    connector = MatrixConnector(
        base_url="http://aios", token="token", spool_path=tmp_path / "answered.sqlite"
    )
    connector.config = config
    store = MemoryASStateStore()
    appservice = create_appservice(config, state_store=store, appservice_class=SupervisedAppService)
    appservice.connector = connector
    appservice.synchronous_handlers = True
    appservice.matrix_event_handler(connector._handle_event)
    connector.az = appservice
    connector.emit_inbound = AsyncMock(return_value={"deduped": False})
    # The homeserver is the authority.  By default it is unreachable; each
    # test states what the homeserver would say, so "what the store knows"
    # and "what is actually true" are always kept distinct.
    appservice._intent = MagicMock()
    appservice._intent.get_joined_members = AsyncMock(
        side_effect=AssertionError("test did not declare the homeserver's answer")
    )
    return connector, appservice, store


def _homeserver_says(appservice, *members: UserID) -> None:
    """Declare the room's TRUE join list, as ``/joined_members`` would return."""
    appservice._intent.get_joined_members = AsyncMock(
        return_value={member: Member(membership=Membership.JOIN) for member in members}
    )


def _message(sender: str = str(HUMAN), room: str = str(ROOM), event_id: str = "$evt1") -> dict:
    return {
        "type": "m.room.message",
        "room_id": room,
        "event_id": event_id,
        "sender": sender,
        "origin_server_ts": 1,
        "content": {"msgtype": "m.text", "body": "hello agent"},
    }


async def _put(appservice, txn_id: str, events: list[dict]):
    async with TestClient(TestServer(appservice.app)) as client:
        return await client.put(
            f"/_matrix/app/v1/transactions/{txn_id}",
            json={"events": events},
            headers={"Authorization": f"Bearer {HS_TOKEN}"},
        )


async def test_partial_view_dm_is_not_acked_and_ignored(receiver) -> None:
    """RED: the store has seen ONLY the human's join; the ghost is LIVE.

    This is the executed loss.  The ghost really is joined and really is
    routable, but its ``m.room.member`` event has not come down a transaction
    yet, so ``get_members`` returns ``[@human]`` -- no namespaced member --
    and the event is classified "genuinely foreign", ACKed 200 and dropped.
    Synapse never retries a transaction it saw succeed, so the DM is gone.

    The property is NOT "must refuse": refusing and delivering are both
    acceptable outcomes.  What is forbidden is the specific pair
    ``ACK + delivered nothing``, which is indistinguishable from success to
    the homeserver and invisible to everyone else.
    """
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    _homeserver_says(appservice, HUMAN, GHOST)
    assert not await store.has_full_member_list(ROOM)

    response = await _put(appservice, "txn-partial-dm", [_message()])

    assert not (response.status < 400 and connector.emit_inbound.await_count == 0), (
        f"receiver ACKed {response.status} having delivered nothing: the store's "
        "partial view made a real DM look foreign, and Synapse will now drop it "
        "permanently"
    )
    # And here the right outcome is the strong one: the membership view was
    # repairable from the homeserver, so the message is DELIVERED, not merely
    # refused.  (A guard that only ever refuses is useless; see also
    # test_complete_view_is_delivered_without_touching_the_homeserver.)
    assert response.status == 200, f"repairable view should deliver, got {response.status}"
    assert connector.emit_inbound.await_args.kwargs["connection_id"] == "con_1"


async def test_partial_view_hiding_an_unroutable_ghost_is_not_acked(receiver) -> None:
    """RED: a partial view bypasses the partial-routability refusal entirely.

    The store has seen 3 of 4 members; the one it has NOT seen is the ghost
    with no live connection -- precisely the recipient that check exists to
    catch.  Every member the store knows about is routable, so the check
    passes and the transaction is ACKed while a recipient's copy is lost.
    """
    connector, appservice, store = receiver
    for member in (HUMAN, HUMAN2, GHOST):
        await store.set_membership(ROOM, member, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    # GHOST2 is really in the room and has NO connection.
    _homeserver_says(appservice, HUMAN, HUMAN2, GHOST, GHOST2)

    response = await _put(appservice, "txn-hidden-unroutable", [_message(event_id="$evt2")])

    assert response.status >= 400, (
        f"receiver ACKed {response.status} for a room whose unseen member is "
        "unroutable; that recipient's copy is silently lost"
    )
    connector.emit_inbound.assert_not_awaited()
    assert "txn-hidden-unroutable" not in appservice.transactions


async def test_unverifiable_view_refuses_rather_than_acking(receiver) -> None:
    """GUARD REFUSES: incomplete view + unreachable homeserver → refuse, not ACK.

    This is the case that makes "incomplete is treated exactly like empty"
    load-bearing: when completeness can be neither confirmed nor repaired,
    the receiver has no idea who the recipients are and must not pretend it
    delivered to them.  It must also stay RETRYABLE -- a homeserver blip is
    not a reason to tear the container down.
    """
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    appservice._intent.get_joined_members = AsyncMock(
        side_effect=ConnectionError("synapse unreachable")
    )

    response = await _put(appservice, "txn-unverifiable", [_message(event_id="$evt3")])

    assert response.status == 503, (
        f"expected a retryable refusal, got {response.status}; an unverifiable "
        "membership view carries no more authority than an empty one"
    )
    connector.emit_inbound.assert_not_awaited()
    assert "txn-unverifiable" not in appservice.transactions
    assert not connector._halt.is_set(), "a transient homeserver failure halted the connector"


async def test_bot_forbidden_repair_uses_a_connected_ghost_intent(receiver) -> None:
    """The bot need not belong to a human↔ghost DM; the ghost does."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    appservice._intent.get_joined_members = AsyncMock(side_effect=MForbidden(403, "not joined"))
    ghost_intent = MagicMock()
    ghost_intent.get_joined_members = AsyncMock(
        return_value={
            HUMAN: Member(membership=Membership.JOIN),
            GHOST: Member(membership=Membership.JOIN),
        }
    )
    appservice._intent.user.return_value = ghost_intent

    response = await _put(appservice, "txn-bot-forbidden-repair", [_message(event_id="$evt7")])

    assert response.status == 200
    connector.emit_inbound.assert_awaited_once()
    appservice._intent.user.assert_called_once_with(GHOST)


async def test_permanently_forbidden_repair_escalates_and_halts(receiver) -> None:
    """A permanent membership refusal gets a finite retry budget, then HALTs loudly."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    forbidden = MForbidden(403, "not joined")
    appservice._intent.get_joined_members = AsyncMock(side_effect=forbidden)
    ghost_intent = MagicMock()
    ghost_intent.get_joined_members = AsyncMock(side_effect=forbidden)
    appservice._intent.user.return_value = ghost_intent

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES):
        response = await _put(appservice, f"txn-forbidden-{attempt}", [_message(event_id="$evt8")])
        statuses.append(response.status)

    assert statuses[:-1] == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES - 1)
    assert statuses[-1] == 500
    assert connector._halt.is_set(), "permanent refusal did not escalate to HALT"
    assert connector._unroutable_attempts["$evt8"] == connector.MAX_UNROUTABLE_REDELIVERIES
    connector.emit_inbound.assert_not_awaited()


async def test_permanent_non_forbidden_repair_failure_escalates_at_bound(receiver) -> None:
    """A terminal 404 must not silently retry forever merely because it is not 403."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    appservice._intent.get_joined_members = AsyncMock(side_effect=MNotFound(404, "room gone"))

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES):
        response = await _put(
            appservice, f"txn-not-found-{attempt}", [_message(event_id="$evt404")]
        )
        statuses.append(response.status)

    assert statuses[:-1] == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES - 1)
    assert statuses[-1] == 500
    assert connector._halt.is_set()
    assert connector._unroutable_attempts["$evt404"] == connector.MAX_UNROUTABLE_REDELIVERIES


async def test_unknown_repair_failure_escalates_at_bound(receiver) -> None:
    """An unclassified repair failure gets a finite budget rather than retrying forever."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    appservice._intent.get_joined_members = AsyncMock(
        side_effect=MatrixResponseError("unexpected homeserver response")
    )

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES):
        response = await _put(
            appservice, f"txn-unknown-{attempt}", [_message(event_id="$evt-unknown")]
        )
        statuses.append(response.status)

    assert statuses[:-1] == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES - 1)
    assert statuses[-1] == 500
    assert connector._halt.is_set()
    assert connector._unroutable_attempts["$evt-unknown"] == (connector.MAX_UNROUTABLE_REDELIVERIES)


async def test_unknown_ghost_repair_failure_escalates_at_bound(receiver) -> None:
    """The bounded default also applies after entering the ghost-candidate path."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    appservice._intent.get_joined_members = AsyncMock(side_effect=MForbidden(403, "not joined"))
    ghost_intent = MagicMock()
    ghost_intent.get_joined_members = AsyncMock(
        side_effect=MatrixResponseError("unexpected homeserver response")
    )
    appservice._intent.user.return_value = ghost_intent

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES):
        response = await _put(
            appservice, f"txn-ghost-unknown-{attempt}", [_message(event_id="$evt-ghost-unknown")]
        )
        statuses.append(response.status)

    assert statuses[:-1] == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES - 1)
    assert statuses[-1] == 500
    assert connector._halt.is_set()
    assert connector._unroutable_attempts["$evt-ghost-unknown"] == (
        connector.MAX_UNROUTABLE_REDELIVERIES
    )


async def test_matrix_server_error_remains_transient_without_consuming_budget(receiver) -> None:
    """A Matrix 5xx is no more permanent than a transport failure."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    appservice._intent.get_joined_members = AsyncMock(side_effect=MUnknown(503, "overloaded"))

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES * 3):
        response = await _put(
            appservice, f"txn-server-error-{attempt}", [_message(event_id="$evt503")]
        )
        statuses.append(response.status)

    assert statuses == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES * 3)
    assert not connector._halt.is_set()
    assert connector._unroutable_attempts == {}


async def test_ghost_repair_is_bounded_and_skips_a_flaky_candidate(receiver) -> None:
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    connector._ghost_connections.update({f"_aios_agent_{i}": f"con_{i}" for i in range(100)})
    appservice._intent.get_joined_members = AsyncMock(side_effect=MForbidden(403, "not joined"))
    ghost_intent = MagicMock()
    ghost_intent.get_joined_members = AsyncMock(
        side_effect=[
            ConnectionError("first ghost is flapping"),
            {HUMAN: Member(membership=Membership.JOIN), GHOST: Member(membership=Membership.JOIN)},
        ]
    )
    appservice._intent.user.return_value = ghost_intent

    response = await _put(appservice, "txn-bounded-repair", [_message(event_id="$evt-bound")])

    assert response.status == 200
    assert ghost_intent.get_joined_members.await_count == 2
    assert ghost_intent.get_joined_members.await_count <= connector.MAX_MEMBERSHIP_REPAIR_CANDIDATES
    assert connector._unroutable_attempts == {}
    for call in ghost_intent.get_joined_members.await_args_list:
        assert call.kwargs["ensure_joined"] is False


async def test_failed_ghost_repair_does_not_probe_every_connected_ghost(receiver) -> None:
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections.update({f"_aios_agent_{i}": f"con_{i}" for i in range(500)})
    forbidden = MForbidden(403, "not joined")
    appservice._intent.get_joined_members = AsyncMock(side_effect=forbidden)
    ghost_intent = MagicMock()
    ghost_intent.get_joined_members = AsyncMock(side_effect=forbidden)
    appservice._intent.user.return_value = ghost_intent

    response = await _put(appservice, "txn-bounded-failure", [_message(event_id="$evt-cap")])

    assert response.status == 503
    assert ghost_intent.get_joined_members.await_count == connector.MAX_MEMBERSHIP_REPAIR_CANDIDATES
    assert ghost_intent.get_joined_members.await_count < len(connector._ghost_connections)


@pytest.mark.parametrize(
    "failure",
    [
        OSError("network is unreachable"),
        socket.gaierror(-3, "Temporary failure in name resolution"),
    ],
    ids=["oserror", "dns-gaierror"],
)
@pytest.mark.parametrize("through_ghost", [False, True], ids=["direct", "ghost"])
async def test_os_network_failures_remain_transient_past_bound(
    receiver, failure: OSError, through_ghost: bool
) -> None:
    """Raw OS/DNS failures are network blips, not reasons to halt the connector."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    failing_intent = appservice._intent
    if through_ghost:
        connector._ghost_connections["_aios_agent_one"] = "con_1"
        appservice._intent.get_joined_members = AsyncMock(side_effect=MForbidden(403, "not joined"))
        failing_intent = MagicMock()
        appservice._intent.user.return_value = failing_intent
    failing_intent.get_joined_members = AsyncMock(side_effect=failure)

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES + 1):
        response = await _put(
            appservice,
            f"txn-os-transient-{through_ghost}-{attempt}",
            [_message(event_id="$evt-os-transient")],
        )
        statuses.append(response.status)

    assert statuses == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES + 1)
    assert not connector._halt.is_set()
    assert connector._unroutable_attempts == {}


async def test_repeated_transient_repair_failure_never_halts(receiver) -> None:
    """Transport failure remains retryable and must not consume the permanent budget."""
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    appservice._intent.get_joined_members = AsyncMock(
        side_effect=ConnectionError("synapse unreachable")
    )

    statuses = []
    for attempt in range(connector.MAX_UNROUTABLE_REDELIVERIES + 1):
        response = await _put(appservice, f"txn-transient-{attempt}", [_message(event_id="$evt9")])
        statuses.append(response.status)

    assert statuses == [503] * (connector.MAX_UNROUTABLE_REDELIVERIES + 1)
    assert not connector._halt.is_set()
    assert connector._unroutable_attempts == {}


async def test_partial_view_does_not_mislabel_a_group_as_a_dm(receiver) -> None:
    """The same unverified list also computes ``room_kind``.

    The store has seen 2 of 3 members, so the room looks like a DM
    (``len(members) == 2``).  Every recipient here IS routable, so this event
    is delivered either way -- what must not happen is delivering it with the
    wrong ``room_kind``, which is a quieter corruption than a drop but comes
    from the same unvalidated input.
    """
    connector, appservice, store = receiver
    await store.set_membership(ROOM, HUMAN, Membership.JOIN)
    await store.set_membership(ROOM, GHOST, Membership.JOIN)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    _homeserver_says(appservice, HUMAN, HUMAN2, GHOST)

    response = await _put(appservice, "txn-room-kind", [_message(event_id="$evt4")])

    assert response.status == 200
    metadata = connector.emit_inbound.await_args.kwargs["metadata"]
    assert metadata["room_kind"] == "group", (
        f"a partly-known 3-person group was labelled {metadata['room_kind']!r}; "
        "room_kind was computed from the same unverified member list"
    )


async def test_complete_view_is_delivered_without_touching_the_homeserver(receiver) -> None:
    """GENUINE PERMIT + no per-event round trip.

    A view the store itself declares complete must be trusted as-is: the
    event is delivered, ACKed 200, and NO ``/joined_members`` call is made.
    Without this, "refuse when unverified" could degrade into either refusing
    everything or issuing a homeserver request for every inbound message.
    """
    connector, appservice, store = receiver
    await store.set_members(
        ROOM,
        {HUMAN: Member(membership=Membership.JOIN), GHOST: Member(membership=Membership.JOIN)},
        only_membership=Membership.JOIN,
    )
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    assert await store.has_full_member_list(ROOM)

    response = await _put(appservice, "txn-complete", [_message(event_id="$evt5")])

    assert response.status == 200, (
        f"a complete, fully routable view was refused ({response.status})"
    )
    connector.emit_inbound.assert_awaited_once()
    assert connector.emit_inbound.await_args.kwargs["connection_id"] == "con_1"
    assert connector.emit_inbound.await_args.kwargs["metadata"]["room_kind"] == "dm"
    appservice._intent.get_joined_members.assert_not_awaited()


async def test_repaired_view_still_acks_genuinely_foreign_rooms(receiver) -> None:
    """OVER-CORRECTION GUARD: repair must not turn foreign rooms into retries.

    A room with no namespaced member is genuinely not ours even after the
    membership view is verified.  It must still ACK, or every unrelated room
    the appservice is told about becomes an unbounded retry storm.
    """
    connector, appservice, store = receiver
    foreign = RoomID("!foreign:your.server")
    await store.set_membership(foreign, HUMAN, Membership.JOIN)
    _homeserver_says(appservice, HUMAN, HUMAN2)

    response = await _put(
        appservice, "txn-foreign-verified", [_message(room=str(foreign), event_id="$evt6")]
    )

    assert response.status == 200, (
        f"receiver refused ({response.status}) a room that is genuinely not ours "
        "even once membership was verified; this is a retry storm"
    )
    connector.emit_inbound.assert_not_awaited()
