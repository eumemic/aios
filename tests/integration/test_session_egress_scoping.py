"""Tenant scoping + fail-closed contract for the live session egress read.

Covers two gaps found in review of the ``GET /v1/sessions/{id}/egress``
build (#2020):

**Cross-account scoping (mutation-verified).** ``get_session_egress``
scopes with ``AND s.account_id = $2``. That predicate was correct but
unexercised: before this file, *deleting it left the entire suite
green*, so the control was indistinguishable from a coincidence.
``test_cross_account_valid_session_id_is_not_found`` is the negative
that turns RED on its removal, and
``test_same_account_read_returns_live_state`` is the positive control
that stops the negative from passing vacuously on a build where the
endpoint returns nothing to anyone.

**Unreadable persisted state.** ``hosts`` is ``JSONB NOT NULL``, but
``NOT NULL`` does not exclude the JSON scalar ``null``, nor a non-array
container, nor an array of objects that do not satisfy
:class:`SessionEgressHost`. Any of those reached the model-validation
loop and surfaced as an unhandled 500. The stated contract is now
fail-closed: unreadable state is reported as absent (``NotFoundError``)
and never partially rendered.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.errors import NotFoundError

pytestmark = pytest.mark.integration

_HOSTS_A: list[dict[str, Any]] = [
    {
        "host": "api.mailgun.com",
        "intercepted": True,
        "source_credential_id": "vcred_a_mailgun",
        "secret_name": "MAILGUN_API_KEY",
    }
]


async def _seed_sessions(conn: asyncpg.Connection[Any]) -> None:
    """One session per tenant, so a cross-account read targets a *valid* id.

    The point of the negative test is that ``sess_a`` genuinely exists
    and genuinely has egress state — the read must be refused because
    of who is asking, not because the id is unresolvable.
    """
    await conn.execute(
        """
        INSERT INTO environments (id, name, account_id)
        VALUES ('env_a', 'env-a', 'acc_a'),
               ('env_b', 'env-b', 'acc_b');
        INSERT INTO agents (id, name, model, account_id)
        VALUES ('agent_a', 'agent-a', 'test/model', 'acc_a'),
               ('agent_b', 'agent-b', 'test/model', 'acc_b');
        INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
        VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_a'),
               ('sess_b', 'agent_b', 'env_b', '/tmp/ws-b', 'acc_b');
        """
    )


async def _stamp_valid(
    conn: asyncpg.Connection[Any], session_id: str, hosts: list[dict[str, Any]]
) -> None:
    await conn.execute(
        "INSERT INTO session_egress_states "
        "(session_id, hosts, provisioned_at, sandbox_generation) "
        "VALUES ($1, $2, now(), 3)",
        session_id,
        hosts,
    )


async def _stamp_raw(conn: asyncpg.Connection[Any], session_id: str, hosts_sql: str) -> None:
    """Persist a hand-written jsonb literal, bypassing the model entirely.

    Simulates state written by an older/other writer — the read path
    cannot assume its own writer produced the row.
    """
    await conn.execute(
        "INSERT INTO session_egress_states "
        "(session_id, hosts, provisioned_at, sandbox_generation) "
        f"VALUES ($1, {hosts_sql}, now(), 3)",
        session_id,
    )


async def test_same_account_read_returns_live_state(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    """POSITIVE CONTROL: the owning tenant still gets its data back.

    Without this, the cross-account negative would also pass on a build
    where the query is broken for everybody.
    """
    conn = conn_two_accounts
    await _seed_sessions(conn)
    await _stamp_valid(conn, "sess_a", _HOSTS_A)

    result = await queries.get_session_egress(conn, "sess_a", account_id="acc_a")

    assert result.sandbox_generation == 3
    assert [host.host for host in result.hosts] == ["api.mailgun.com"]
    assert result.hosts[0].source_credential_id == "vcred_a_mailgun"
    assert result.hosts[0].secret_name == "MAILGUN_API_KEY"
    assert result.hosts[0].intercepted is True


async def test_cross_account_valid_session_id_is_not_found(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    """Account B asking for account A's real session id gets the not-found path.

    MUTATION CONTROL: deleting ``AND s.account_id = $2`` from
    ``get_session_egress`` makes this test fail (the row is returned
    instead of raising).
    """
    conn = conn_two_accounts
    await _seed_sessions(conn)
    await _stamp_valid(conn, "sess_a", _HOSTS_A)

    with pytest.raises(NotFoundError) as excinfo:
        await queries.get_session_egress(conn, "sess_a", account_id="acc_b")

    # Refusal must not become a side channel for the data it withheld.
    rendered = excinfo.value.to_message()
    assert "api.mailgun.com" not in rendered
    assert "vcred_a_mailgun" not in rendered
    assert "MAILGUN_API_KEY" not in rendered


async def test_cross_account_read_is_refused_even_with_own_state_present(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    """B holding its own egress state must not make A's row readable.

    Guards the ``JOIN`` shape specifically: a scoping bug that matched
    on *any* row owned by the caller would pass the simpler negative.
    """
    conn = conn_two_accounts
    await _seed_sessions(conn)
    await _stamp_valid(conn, "sess_a", _HOSTS_A)
    await _stamp_valid(
        conn,
        "sess_b",
        [
            {
                "host": "api.stripe.com",
                "intercepted": True,
                "source_credential_id": "vcred_b_stripe",
                "secret_name": "STRIPE_API_KEY",
            }
        ],
    )

    with pytest.raises(NotFoundError):
        await queries.get_session_egress(conn, "sess_a", account_id="acc_b")

    # ...and B's own read still works, so the refusal above is scoping,
    # not collateral breakage.
    mine = await queries.get_session_egress(conn, "sess_b", account_id="acc_b")
    assert [host.host for host in mine.hosts] == ["api.stripe.com"]


@pytest.mark.parametrize(
    ("label", "hosts_sql"),
    [
        # ``NOT NULL`` does not exclude the JSON scalar null.
        ("json_null", "'null'::jsonb"),
        ("json_scalar", "'5'::jsonb"),
        ("object_not_array", """'{"host": "api.mailgun.com"}'::jsonb"""),
        ("array_of_scalars", """'["api.mailgun.com"]'::jsonb"""),
        ("entry_missing_required_fields", """'[{"host": "api.mailgun.com"}]'::jsonb"""),
        # NB: pydantic lax mode coerces the bool-ish strings ("yes"/"no"/
        # "true"), so those are NOT corrupt input; "maybe" is the genuinely
        # uncoercible case. Verified empirically, not assumed.
        (
            "entry_wrong_type",
            """'[{"host": "a.com", "intercepted": "maybe", """
            """"source_credential_id": "v", "secret_name": "S"}]'::jsonb""",
        ),
        ("entry_null", "'[null]'::jsonb"),
    ],
)
async def test_unreadable_persisted_state_fails_closed(
    conn_two_accounts: asyncpg.Connection[Any], label: str, hosts_sql: str
) -> None:
    """Unreadable persisted state is reported as absent, not as a 500.

    Each variant previously escaped as an unhandled ``TypeError`` or
    pydantic ``ValidationError`` out of the model-validation loop.
    """
    conn = conn_two_accounts
    await _seed_sessions(conn)
    await _stamp_raw(conn, "sess_a", hosts_sql)

    with pytest.raises(NotFoundError):
        await queries.get_session_egress(conn, "sess_a", account_id="acc_a")


async def test_unknown_persisted_keys_are_not_rendered(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    """Unknown keys in persisted state are dropped, not echoed to the caller.

    The projection to :class:`SessionEgressHost` currently relies on
    pydantic's DEFAULT ``extra="ignore"``: the model declares no
    ``model_config``, so unknown keys are silently discarded. That is the
    behaviour the endpoint's safety rests on, and before this test nothing
    pinned it — flipping the model to ``extra="allow"`` (or swapping the
    projection for a passthrough) would start echoing whatever the writer
    put in the row, with no failing test.

    This pins the OBSERVABLE guarantee (unknown persisted keys never reach
    the response) without asserting *how* it is achieved. It is a
    regression guard, NOT a redaction boundary for the four DECLARED
    fields — see the deferred write-up on the PR.
    """
    conn = conn_two_accounts
    await _seed_sessions(conn)
    await _stamp_raw(
        conn,
        "sess_a",
        """'[{"host": "api.mailgun.com", "intercepted": true, """
        """"source_credential_id": "vcred_a_mailgun", "secret_name": "MAILGUN_API_KEY", """
        """"secret_value": "sk-live-must-not-appear", "internal_ip": "10.0.0.5"}]'::jsonb""",
    )

    result = await queries.get_session_egress(conn, "sess_a", account_id="acc_a")

    rendered = result.model_dump_json()
    assert "sk-live-must-not-appear" not in rendered
    assert "10.0.0.5" not in rendered
    # The declared fields still come through — this is not a broken read.
    assert result.hosts[0].host == "api.mailgun.com"
    assert result.hosts[0].secret_name == "MAILGUN_API_KEY"


async def test_unreadable_state_does_not_leak_raw_payload(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    """The fail-closed error must not echo the unparsable payload back."""
    conn = conn_two_accounts
    await _seed_sessions(conn)
    await _stamp_raw(
        conn,
        "sess_a",
        """'[{"host": "internal-10-0-0-5.corp", "leaked_secret": "sk-live-abc123"}]'::jsonb""",
    )

    with pytest.raises(NotFoundError) as excinfo:
        await queries.get_session_egress(conn, "sess_a", account_id="acc_a")

    rendered = excinfo.value.to_message()
    assert "sk-live-abc123" not in rendered
    assert "internal-10-0-0-5.corp" not in rendered
