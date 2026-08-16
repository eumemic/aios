"""Inbound routing must FAIL CLOSED when routing state is not ready.

The Matrix appservice transaction endpoint is an at-least-once delivery
channel: Synapse retries a transaction until it gets a 2xx, and *permanently
drops it* once it does.  So a 200 is a promise that the events in the
transaction have been durably taken over.

``MatrixConnector.setup()`` starts the appservice HTTP listener before the
discovery loop has necessarily populated ``_ghost_connections``.  An inbound
message arriving in that window addresses a ghost we own (it is inside our
namespace) but that we cannot yet route.  Answering 200 there converts a
transient not-ready condition into permanent message loss.

Two facts the receiver must keep apart:

* **genuinely foreign** — no namespaced member in the room; the event is not
  ours and never will be.  ACK 2xx and ignore, or we induce a retry storm on
  every unrelated room the appservice is told about.
* **not routable yet** — a namespaced member is present but has no live
  connection.  We cannot tell whether it is unroutable forever or merely
  unroutable right now, so we must NOT ACK.

These tests drive the real aiohttp transaction endpoint rather than calling
``_handle_event`` directly, because the defect is precisely in the status code
the HTTP layer returns.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from mautrix.appservice.state_store.memory import ASStateStore
from mautrix.client.state_store.memory import MemoryStateStore
from mautrix.types import Member, Membership, RoomID, UserID

from aios_matrix.appservice import create_appservice
from aios_matrix.config import MatrixConfig
from aios_matrix.connector import MatrixConnector, SupervisedAppService

HS_TOKEN = "hs-secret"
ROOM = RoomID("!dm:your.server")
GHOST = UserID("@_aios_agent_one:your.server")
HUMAN = UserID("@human:your.server")


class MemoryASStateStore(MemoryStateStore, ASStateStore):
    """In-memory state store with the appservice mixin, no Postgres."""

    def __init__(self) -> None:
        MemoryStateStore.__init__(self)
        ASStateStore.__init__(self)


async def set_known_room(store, room: RoomID, members: tuple[UserID, ...]) -> None:
    """Record a room whose join list is KNOWN COMPLETE.

    Deliberately ``set_members`` rather than repeated ``set_membership``:
    only the bulk write marks ``has_full_member_list``, which is exactly the
    distinction the connector now checks.  ``set_membership`` models the
    incremental timeline learning that leaves the view PARTIAL, and tests
    that want a complete view must not accidentally build a partial one --
    otherwise every test in this file would exercise the refuse path and the
    guard would look correct while permitting nothing.
    """
    await store.set_members(
        room,
        {member: Member(membership=Membership.JOIN) for member in members},
        only_membership=Membership.JOIN,
    )


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
    """A connector wired to a live appservice app, with NO connections served.

    This reproduces the startup window: the HTTP listener is up (Synapse can
    deliver) but ``_ghost_connections`` is still empty because no
    ``serve_connection`` worker has run yet.
    """
    connector = MatrixConnector(
        base_url="http://aios", token="token", spool_path=tmp_path / "answered.sqlite"
    )
    connector.config = config
    state_store = MemoryASStateStore()
    appservice = create_appservice(
        config, state_store=state_store, appservice_class=SupervisedAppService
    )
    appservice.connector = connector
    appservice.synchronous_handlers = True
    appservice.matrix_event_handler(connector._handle_event)
    connector.az = appservice
    # The ghost IS in the room and IS in our namespace — we own it.  What we
    # do not have (yet) is the connection_id needed to route its traffic.
    await set_known_room(state_store, ROOM, (GHOST, HUMAN))
    assert connector._ghost_connections == {}
    connector.emit_inbound = AsyncMock(return_value={"deduped": False})
    return connector, appservice


def _txn(events: list[dict]) -> dict:
    return {"events": events}


def _message(sender: str = str(HUMAN), room: str = str(ROOM)) -> dict:
    return {
        "type": "m.room.message",
        "room_id": room,
        "event_id": "$evt1",
        "sender": sender,
        "origin_server_ts": 1,
        "content": {"msgtype": "m.text", "body": "hello agent"},
    }


async def _put(appservice, txn_id: str, body: dict):
    async with TestClient(TestServer(appservice.app)) as client:
        return await client.put(
            f"/_matrix/app/v1/transactions/{txn_id}",
            json=body,
            headers={"Authorization": f"Bearer {HS_TOKEN}"},
        )


async def test_unroutable_namespaced_event_is_not_acked(receiver) -> None:
    """RED: a DM to one of our ghosts, arriving before routing state is ready.

    Pins the delivery-safety property: the receiver must not claim delivery of
    a message it did not deliver.  A 2xx here is silent, permanent loss of a
    human's DM — Synapse never retries a transaction it saw succeed.
    """
    connector, appservice = receiver

    response = await _put(appservice, "txn-startup-race", _txn([_message()]))

    assert response.status >= 400, (
        f"receiver ACKed {response.status} for an event it could not route; "
        "Synapse will treat this message as delivered and never retry it"
    )
    # And it must not have been silently swallowed as if handled.
    connector.emit_inbound.assert_not_awaited()
    # The transaction must remain un-recorded so the retry is processed, not
    # short-circuited as a duplicate by ``_read_transaction_header``.
    assert "txn-startup-race" not in appservice.transactions


async def test_foreign_event_is_still_acked(receiver) -> None:
    """REGRESSION / over-correction guard.

    A room with no namespaced member is genuinely not ours.  Retrying it would
    never succeed, so failing closed here would turn every unrelated room into
    an unbounded retry storm against the homeserver.  It must ACK and ignore.
    """
    connector, appservice = receiver
    other_room = RoomID("!foreign:your.server")
    await set_known_room(
        appservice.state_store, other_room, (HUMAN, UserID("@someone_else:your.server"))
    )

    response = await _put(appservice, "txn-foreign", _txn([_message(room=str(other_room))]))

    assert response.status == 200, (
        f"receiver refused ({response.status}) an event that is genuinely not "
        "ours; this induces an unbounded Synapse retry storm"
    )
    connector.emit_inbound.assert_not_awaited()


async def test_routable_event_is_acked_and_delivered(receiver) -> None:
    """The happy path stays intact once routing state IS populated."""
    connector, appservice = receiver
    connector._ghost_connections["_aios_agent_one"] = "con_1"

    response = await _put(appservice, "txn-live", _txn([_message()]))

    assert response.status == 200
    connector.emit_inbound.assert_awaited_once()
    assert connector.emit_inbound.await_args.kwargs["connection_id"] == "con_1"


async def test_own_ghost_echo_is_acked_not_retried(receiver) -> None:
    """Over-correction guard #2: our own ghost's message must not fail closed.

    When the sender IS the namespaced ghost, there is no *other* receiver in a
    DM, so a naive ``namespaced and not routable -> retry`` rule would fail
    closed on every message the agent itself sends — an unbounded retry loop
    on our own outbound echo, and one that persists even after routing state
    is fully populated.  A ghost's own echo is not undeliverable; it is simply
    not inbound traffic.
    """
    connector, appservice = receiver

    response = await _put(appservice, "txn-echo", _txn([_message(sender=str(GHOST))]))

    assert response.status == 200, (
        f"receiver refused ({response.status}) its own ghost's echo; this "
        "retries forever and never becomes routable"
    )
    connector.emit_inbound.assert_not_awaited()


async def test_empty_membership_state_is_not_acked(receiver) -> None:
    """A room we have no state for is unclassifiable, so it must not ACK.

    ``get_members`` returning empty cannot mean "empty room" — a room always
    contains at least the sender of the event being delivered.  It means the
    state store has nothing for this room yet, which is exactly the cold-start
    window.  Classifying that as "not ours" silently drops a real DM.
    """
    connector, appservice = receiver
    unknown_room = RoomID("!nostate:your.server")

    response = await _put(appservice, "txn-nostate", _txn([_message(room=str(unknown_room))]))

    assert response.status >= 400, (
        f"receiver ACKed {response.status} for a room it has no state for; "
        "it cannot know the message was not ours"
    )
    connector.emit_inbound.assert_not_awaited()


async def test_routing_not_ready_is_retryable_not_halt(receiver) -> None:
    """The failure must be classified RETRY, not HALT.

    ``_classify_receiver_failure`` defaults unknown exceptions to HALT, which
    trips ``_halt`` and tears the container down.  A routine startup race must
    not be a container-killing event, and must return a retryable 5xx.
    """
    connector, appservice = receiver

    response = await _put(appservice, "txn-race-status", _txn([_message()]))

    assert response.status == 503, f"expected retryable 503, got {response.status}"
    assert not connector._halt.is_set(), (
        "a routine startup race halted the receiver; this converts a transient "
        "condition into a container-level outage"
    )


async def test_non_message_event_is_acked(receiver) -> None:
    """A non-message event in an owned room is not inbound traffic: ACK it."""
    connector, appservice = receiver

    response = await _put(
        appservice,
        "txn-typing",
        _txn(
            [
                {
                    "type": "m.room.topic",
                    "room_id": str(ROOM),
                    "event_id": "$topic",
                    "sender": str(HUMAN),
                    "origin_server_ts": 1,
                    "state_key": "",
                    "content": {"topic": "hi"},
                }
            ]
        ),
    )

    assert response.status == 200
    connector.emit_inbound.assert_not_awaited()


async def test_supervised_handler_is_actually_bound_to_the_route(config) -> None:
    """The override must be REACHABLE, not merely defined.

    ``AppService.__init__`` calls ``register_routes()``, which binds
    ``self._http_handle_transaction`` into the aiohttp router at construction
    time.  Installing the subclass afterwards (by reassigning ``__class__``)
    leaves the ORIGINAL bound method serving requests, so the entire
    supervised ACK/RETRY/HALT receiver is dead code and every failure falls
    through to mautrix's catch-all ``return web.json_response(output or {})``
    -- a 200.  This pins the wiring itself, which no behavioural test of a
    correctly-constructed appservice would catch.
    """
    appservice = create_appservice(
        config, state_store=MemoryASStateStore(), appservice_class=SupervisedAppService
    )
    handlers = [
        route.handler
        for route in appservice.app.router.routes()
        if "transactions" in str(route.resource.canonical if route.resource else "")
    ]
    assert handlers, "no transaction route registered"
    for handler in handlers:
        assert (
            getattr(handler, "__func__", None) is SupervisedAppService._http_handle_transaction
        ), (
            f"transaction route is bound to {handler.__qualname__}, not the "
            "supervised override; the ACK/RETRY/HALT logic is unreachable"
        )


# ─── mixed-routability: "some ghosts present" ≠ "all ghosts routable" ─────────

GHOST2 = UserID("@_aios_agent_two:your.server")


async def _join_second_ghost(appservice) -> None:
    await set_known_room(appservice.state_store, ROOM, (GHOST, HUMAN, GHOST2))


async def test_mixed_live_and_unroutable_room_is_not_acked(receiver) -> None:
    """RED: a group room with one LIVE ghost and one not-yet-routable ghost.

    ``ghosts`` is non-empty, so the pre-fix code emits to the live subset,
    returns normally, and the HTTP layer ACKs 200 -- permanently losing the
    second recipient's copy.  "Some ghosts present" is not "all ghosts
    routable".
    """
    connector, appservice = receiver
    await _join_second_ghost(appservice)
    connector._ghost_connections["_aios_agent_one"] = "con_1"

    response = await _put(appservice, "txn-mixed", _txn([_message()]))

    assert response.status >= 400, (
        f"receiver ACKed {response.status} for a room where only SOME "
        "recipients were routable; the unroutable recipient's copy is lost"
    )
    assert "txn-mixed" not in appservice.transactions


async def test_all_live_group_room_delivers_to_every_ghost_and_acks(receiver) -> None:
    """POSITIVE CONTROL: an all-live group room still delivers AND ACKs.

    Without this, "we never ACK a partial delivery" would pass trivially on
    a build that never delivers anything at all.
    """
    connector, appservice = receiver
    await _join_second_ghost(appservice)
    connector._ghost_connections["_aios_agent_one"] = "con_1"
    connector._ghost_connections["_aios_agent_two"] = "con_2"

    response = await _put(appservice, "txn-all-live", _txn([_message()]))

    assert response.status == 200, f"all-live room was refused ({response.status})"
    delivered = {call.kwargs["connection_id"] for call in connector.emit_inbound.await_args_list}
    assert delivered == {"con_1", "con_2"}, f"delivered to {delivered}, expected both ghosts"


async def test_partial_delivery_emits_nothing_at_all(receiver) -> None:
    """A partial-routability refusal must not half-deliver.

    Pins the atomicity choice: the live subset is NOT emitted to before the
    refusal, so the retry does not depend on emit idempotency to be correct.
    """
    connector, appservice = receiver
    await _join_second_ghost(appservice)
    connector._ghost_connections["_aios_agent_one"] = "con_1"

    await _put(appservice, "txn-mixed-noemit", _txn([_message()]))

    connector.emit_inbound.assert_not_awaited()


async def test_repeated_unroutable_redelivery_escalates_to_halt(receiver) -> None:
    """SECOND FINDING: a permanently-absent recipient must stop retrying forever.

    Synapse marks the whole appservice DOWN on a non-2xx and retries the
    OLDEST unsent transaction with 2**n backoff before delivering anything
    newer -- so an eternally-503ing event head-of-line blocks every Matrix
    transaction.  After a bounded number of redeliveries the condition must
    escalate to a loud HALT rather than looping invisibly.

    HALT still answers non-2xx: the message is not dropped, it is escalated.
    """
    connector, appservice = receiver
    connector.MAX_UNROUTABLE_REDELIVERIES = 3

    for attempt in range(2):
        response = await _put(appservice, f"txn-stuck-{attempt}", _txn([_message()]))
        assert response.status == 503, f"attempt {attempt} gave {response.status}, want 503"
        assert not connector._halt.is_set(), "halted before the retry budget was spent"

    final = await _put(appservice, "txn-stuck-final", _txn([_message()]))

    assert final.status >= 400, "escalation must NOT ack; the message is not deliverable"
    assert final.status == 500, f"expected non-retryable 500 on escalation, got {final.status}"
    assert connector._halt.is_set(), "exhausted retry budget did not escalate loudly"
    connector.emit_inbound.assert_not_awaited()


async def test_recovered_routing_clears_the_escalation_counter(receiver) -> None:
    """A stuck event that becomes routable must not stay near the halt edge.

    Without the reset, a room that flaps unroutable/routable would eventually
    halt the container on a fully-delivered message.
    """
    connector, appservice = receiver
    connector.MAX_UNROUTABLE_REDELIVERIES = 3

    stuck = await _put(appservice, "txn-flap-1", _txn([_message()]))
    assert stuck.status == 503
    assert connector._unroutable_attempts

    connector._ghost_connections["_aios_agent_one"] = "con_1"
    recovered = await _put(appservice, "txn-flap-2", _txn([_message()]))

    assert recovered.status == 200
    connector.emit_inbound.assert_awaited_once()
    assert connector._unroutable_attempts == {}, "attempt ledger not cleared after delivery"
