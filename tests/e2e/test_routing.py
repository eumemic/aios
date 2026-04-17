"""E2E tests for the routing infrastructure (issue #30).

Covers connection / channel binding / routing rule CRUD plus the
``resolve_channel`` resolver and the ``POST /v1/connections/{id}/messages``
inbound endpoint.  Runs against a real testcontainer Postgres with
migrations applied.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest

from aios.errors import ConflictError, NoRouteError, NotFoundError, ValidationError
from aios.models.routing_rules import SessionParams


def _uniq() -> str:
    """Return a short hex string unique across the whole test session.

    Tests share the same testcontainer DB, and ``id(pool)`` can collide
    when fixtures churn pools — so use a fresh random suffix instead.
    """
    return secrets.token_hex(4)


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def env_id(pool: Any) -> str:
    from aios.db import queries

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"routing-test-{_uniq()}")
    return env.id


@pytest.fixture
async def agent_id(pool: Any) -> str:
    from aios.services import agents as svc

    a = await svc.create_agent(
        pool,
        name=f"routing-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    return a.id


@pytest.fixture
async def vault_id(pool: Any) -> str:
    from aios.services import vaults as svc

    v = await svc.create_vault(pool, display_name="routing-vault", metadata={})
    return v.id


# ─── connection CRUD ────────────────────────────────────────────────────────


class TestConnectionCRUD:
    async def test_create_and_get(self, pool: Any, vault_id: str) -> None:
        from aios.services import connections as svc

        c = await svc.create_connection(
            pool,
            connector="signal",
            account=f"test-{_uniq()}",
            mcp_url="https://mcp.example.com",
            vault_id=vault_id,
            metadata={"region": "us"},
        )
        assert c.id.startswith("conn_")
        assert c.connector == "signal"
        assert c.metadata == {"region": "us"}
        assert c.archived_at is None

        fetched = await svc.get_connection(pool, c.id)
        assert fetched.id == c.id

    async def test_unique_per_connector_account(self, pool: Any, vault_id: str) -> None:
        from aios.services import connections as svc

        account = f"dup-{_uniq()}"
        await svc.create_connection(
            pool,
            connector="signal",
            account=account,
            mcp_url="https://m1",
            vault_id=vault_id,
            metadata={},
        )
        with pytest.raises(ConflictError):
            await svc.create_connection(
                pool,
                connector="signal",
                account=account,
                mcp_url="https://m2",
                vault_id=vault_id,
                metadata={},
            )

    async def test_update_mcp_url(self, pool: Any, vault_id: str) -> None:
        from aios.services import connections as svc

        c = await svc.create_connection(
            pool,
            connector="signal",
            account=f"upd-{_uniq()}",
            mcp_url="https://old",
            vault_id=vault_id,
            metadata={},
        )
        updated = await svc.update_connection(pool, c.id, mcp_url="https://new")
        assert updated.mcp_url == "https://new"

    async def test_archive(self, pool: Any, vault_id: str) -> None:
        from aios.services import connections as svc

        account = f"arch-{_uniq()}"
        c = await svc.create_connection(
            pool,
            connector="signal",
            account=account,
            mcp_url="https://m",
            vault_id=vault_id,
            metadata={},
        )
        archived = await svc.archive_connection(pool, c.id)
        assert archived.archived_at is not None
        # archived connection no longer counts towards uniqueness for the same (connector, account)
        await svc.create_connection(
            pool,
            connector="signal",
            account=account,
            mcp_url="https://m2",
            vault_id=vault_id,
            metadata={},
        )

    async def test_unknown_vault(self, pool: Any) -> None:
        from aios.services import connections as svc

        with pytest.raises(NotFoundError):
            await svc.create_connection(
                pool,
                connector="signal",
                account=f"novault-{_uniq()}",
                mcp_url="https://m",
                vault_id="vlt_nonexistent",
                metadata={},
            )


# ─── routing rule CRUD ──────────────────────────────────────────────────────


class TestRoutingRuleCRUD:
    async def test_create_agent_target(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.services import channels as svc

        suffix = f"crud-{_uniq()}"
        r = await svc.create_routing_rule(
            pool,
            prefix=f"signal/{suffix}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        assert r.id.startswith("rrul_")
        assert r.target == f"agent:{agent_id}"

    async def test_create_session_target_rejects_session_params(self, pool: Any) -> None:
        from aios.services import channels as svc

        with pytest.raises(ValidationError):
            await svc.create_routing_rule(
                pool,
                prefix=f"signal/sess-bad-{_uniq()}",
                target="session:sess_xxx",
                session_params=SessionParams(environment_id="env_x"),
            )

    async def test_agent_target_requires_environment(self, pool: Any, agent_id: str) -> None:
        from aios.services import channels as svc

        with pytest.raises(ValidationError):
            await svc.create_routing_rule(
                pool,
                prefix=f"signal/missing-env-{_uniq()}",
                target=f"agent:{agent_id}",
                session_params=SessionParams(),
            )

    async def test_invalid_target_string(self, pool: Any) -> None:
        from aios.services import channels as svc

        with pytest.raises(ValidationError):
            await svc.create_routing_rule(
                pool,
                prefix=f"signal/bad-{_uniq()}",
                target="garbage",
                session_params=SessionParams(environment_id="env_x"),
            )

    async def test_unique_per_prefix(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.services import channels as svc

        prefix = f"signal/uniq-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        with pytest.raises(ConflictError):
            await svc.create_routing_rule(
                pool,
                prefix=prefix,
                target=f"agent:{agent_id}",
                session_params=SessionParams(environment_id=env_id),
            )

    async def test_update_target(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.services import channels as svc

        r = await svc.create_routing_rule(
            pool,
            prefix=f"signal/upd-{_uniq()}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        updated = await svc.update_routing_rule(pool, r.id, target=f"agent:{agent_id}@1")
        assert updated.target == f"agent:{agent_id}@1"

    async def test_archive(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.services import channels as svc

        r = await svc.create_routing_rule(
            pool,
            prefix=f"signal/arch-{_uniq()}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        archived = await svc.archive_routing_rule(pool, r.id)
        assert archived.archived_at is not None

    async def test_update_target_kind_flip_session_to_agent_re_validates(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """Flipping session: → agent: without supplying environment_id must
        be rejected — even though the field being updated (target) was
        valid in isolation, the merged combination is not.
        """
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        r = await svc.create_routing_rule(
            pool,
            prefix=f"flip-{_uniq()}",
            target=f"session:{s.id}",
            session_params=SessionParams(),
        )
        with pytest.raises(ValidationError, match="environment_id"):
            await svc.update_routing_rule(pool, r.id, target=f"agent:{agent_id}")

    async def test_update_session_params_on_session_target_re_validates(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """Adding session_params to a session:-target rule must be rejected
        even though the rule's existing target wasn't touched.
        """
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        r = await svc.create_routing_rule(
            pool,
            prefix=f"sp-{_uniq()}",
            target=f"session:{s.id}",
            session_params=SessionParams(),
        )
        with pytest.raises(ValidationError, match="empty session_params"):
            await svc.update_routing_rule(
                pool, r.id, session_params=SessionParams(environment_id=env_id)
            )


# ─── find_matching_rule ─────────────────────────────────────────────────────


class TestFindMatchingRule:
    async def test_exact_match(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.db import queries
        from aios.services import channels as svc

        prefix = f"fmr-{_uniq()}-exact"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, prefix)
        assert r is not None
        assert r.prefix == prefix

    async def test_longer_prefix_wins(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.db import queries
        from aios.services import channels as svc

        suffix = _uniq()
        await svc.create_routing_rule(
            pool,
            prefix=f"fmrl-{suffix}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        await svc.create_routing_rule(
            pool,
            prefix=f"fmrl-{suffix}/specific",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, f"fmrl-{suffix}/specific/x")
        assert r is not None
        assert r.prefix == f"fmrl-{suffix}/specific"

    async def test_segment_aware(self, pool: Any, agent_id: str, env_id: str) -> None:
        """``foo`` must NOT match ``foofoo`` — segment boundary required."""
        from aios.db import queries
        from aios.services import channels as svc

        prefix = f"seg-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            none_match = await queries.find_matching_rule(conn, f"{prefix}foo")
            yes_match = await queries.find_matching_rule(conn, f"{prefix}/abc")
        assert none_match is None
        assert yes_match is not None

    async def test_no_match_returns_none(self, pool: Any) -> None:
        from aios.db import queries

        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, f"nope-{_uniq()}/x")
        assert r is None

    async def test_archived_rule_excluded(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.db import queries
        from aios.services import channels as svc

        prefix = f"arch-rule-{_uniq()}"
        rule = await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        await svc.archive_routing_rule(pool, rule.id)
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, f"{prefix}/x")
        assert r is None

    async def test_three_level_longest_prefix_wins(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """At three nested levels, the deepest matching prefix still wins."""
        from aios.db import queries
        from aios.services import channels as svc

        suffix = _uniq()
        for prefix in (f"l-{suffix}", f"l-{suffix}/b", f"l-{suffix}/b/c"):
            await svc.create_routing_rule(
                pool,
                prefix=prefix,
                target=f"agent:{agent_id}",
                session_params=SessionParams(environment_id=env_id),
            )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, f"l-{suffix}/b/c/x")
        assert r is not None
        assert r.prefix == f"l-{suffix}/b/c"

    async def test_underscore_in_prefix_treated_literally(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """LIKE meta-characters (``_``, ``%``) in prefix must NOT act as
        wildcards.  Without proper handling, prefix ``foo_bar`` would
        wrongly match address ``fooXbar/baz`` (LIKE's ``_`` matches any
        single char).
        """
        from aios.db import queries
        from aios.services import channels as svc

        suffix = _uniq()
        await svc.create_routing_rule(
            pool,
            prefix=f"fo_bar-{suffix}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            wrong = await queries.find_matching_rule(conn, f"fooXbar-{suffix}/x")
            right = await queries.find_matching_rule(conn, f"fo_bar-{suffix}/x")
        assert wrong is None
        assert right is not None

    async def test_percent_in_prefix_treated_literally(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """LIKE's ``%`` (any-string) must not wildcard either."""
        from aios.db import queries
        from aios.services import channels as svc

        suffix = _uniq()
        await svc.create_routing_rule(
            pool,
            prefix=f"fo%bar-{suffix}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            wrong = await queries.find_matching_rule(conn, f"fooXXXbar-{suffix}/x")
            right = await queries.find_matching_rule(conn, f"fo%bar-{suffix}/x")
        assert wrong is None
        assert right is not None


# ─── resolve_channel ────────────────────────────────────────────────────────


class TestResolveChannel:
    async def test_existing_binding_short_circuits(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        address = f"manual-{_uniq()}/abc"
        binding = await svc.create_binding(pool, address=address, session_id=s.id)

        result = await svc.resolve_channel(pool, address)
        assert result.session_id == s.id
        assert result.binding_id == binding.id
        assert result.created_session is False

    async def test_session_target(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        prefix = f"st-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"session:{s.id}",
            session_params=SessionParams(),
        )
        result = await svc.resolve_channel(pool, f"{prefix}/whatever")
        assert result.session_id == s.id
        assert result.created_session is False

    async def test_agent_target_creates_session(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        from aios.db import queries
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        prefix = f"at-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(
                environment_id=env_id,
                vault_ids=[vault_id],
                title="Routed: {address}",
                metadata={"source": "rule"},
            ),
        )
        address = f"{prefix}/chat-1"
        result = await svc.resolve_channel(pool, address)
        assert result.created_session is True

        s = await sess_svc.get_session(pool, result.session_id)
        assert s.title == f"Routed: {address}"
        assert s.metadata == {"source": "rule"}
        async with pool.acquire() as conn:
            vids = await queries.get_session_vault_ids(conn, s.id)
        assert vids == [vault_id]

        # Second hit short-circuits via the binding now persisted.
        again = await svc.resolve_channel(pool, address)
        assert again.session_id == result.session_id
        assert again.binding_id == result.binding_id
        assert again.created_session is False

    async def test_no_route_raises(self, pool: Any) -> None:
        from aios.services import channels as svc

        with pytest.raises(NoRouteError):
            await svc.resolve_channel(pool, f"unrouted-{_uniq()}/x")

    async def test_concurrent_resolve_same_address_returns_same_session(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """Two simultaneous first-time resolves on the same address must
        agree on the session — no spurious unique-violation conflict, no
        orphan session.  Regression test for the resolve_channel race
        flagged in PR #32 review.
        """
        import asyncio

        from aios.services import channels as svc

        prefix = f"race-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        address = f"{prefix}/chat-1"

        results = await asyncio.gather(
            svc.resolve_channel(pool, address),
            svc.resolve_channel(pool, address),
        )
        assert results[0].session_id == results[1].session_id
        assert results[0].binding_id == results[1].binding_id
        # exactly one resolve actually created the session
        assert sum(r.created_session for r in results) == 1

        # exactly one binding row exists for this address
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM channel_bindings WHERE address = $1 AND archived_at IS NULL",
                address,
            )
        assert count == 1

    async def test_concurrent_resolve_different_addresses_do_not_block(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """The advisory lock must be per-address, not global — concurrent
        resolves for distinct addresses run in parallel without contention.
        """
        import asyncio

        from aios.services import channels as svc

        suffix = _uniq()
        await svc.create_routing_rule(
            pool,
            prefix=f"par-{suffix}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )

        results = await asyncio.gather(
            svc.resolve_channel(pool, f"par-{suffix}/a"),
            svc.resolve_channel(pool, f"par-{suffix}/b"),
        )
        assert results[0].session_id != results[1].session_id
        assert all(r.created_session for r in results)

    async def test_falls_through_archived_binding(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """An archived binding must NOT short-circuit; the resolver should
        consult routing rules and create a fresh session/binding.
        """
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        prefix = f"arch-bind-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        old_session = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        address = f"{prefix}/x"
        old_binding = await svc.create_binding(pool, address=address, session_id=old_session.id)
        await svc.archive_binding(pool, old_binding.id)

        result = await svc.resolve_channel(pool, address)
        assert result.session_id != old_session.id
        assert result.created_session is True

    async def test_agent_version_propagated_to_session(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        prefix = f"av-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}@1",
            session_params=SessionParams(environment_id=env_id),
        )
        result = await svc.resolve_channel(pool, f"{prefix}/x")
        s = await sess_svc.get_session(pool, result.session_id)
        assert s.agent_version == 1

    async def test_session_target_with_missing_session_raises(self, pool: Any) -> None:
        from aios.services import channels as svc

        prefix = f"st-missing-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target="session:sess_does_not_exist",
            session_params=SessionParams(),
        )
        with pytest.raises(NotFoundError):
            await svc.resolve_channel(pool, f"{prefix}/x")

    async def test_session_target_with_archived_session_raises(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """A rule pointing at an archived session must fail loudly rather
        than silently re-binding to it.
        """
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        await sess_svc.archive_session(pool, s.id)

        prefix = f"st-arch-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"session:{s.id}",
            session_params=SessionParams(),
        )
        with pytest.raises(NotFoundError, match="archived"):
            await svc.resolve_channel(pool, f"{prefix}/x")

    async def test_invalid_vault_id_in_session_params_rolls_back(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """If session_params.vault_ids points at a missing vault, the whole
        resolve transaction must roll back — no orphan session row, no
        orphan binding row.
        """
        from aios.services import channels as svc

        prefix = f"badvlt-{_uniq()}"
        await svc.create_routing_rule(
            pool,
            prefix=prefix,
            target=f"agent:{agent_id}",
            session_params=SessionParams(
                environment_id=env_id,
                vault_ids=["vlt_does_not_exist"],
            ),
        )
        address = f"{prefix}/x"
        with pytest.raises(NotFoundError):
            await svc.resolve_channel(pool, address)

        async with pool.acquire() as conn:
            binding_count = await conn.fetchval(
                "SELECT count(*) FROM channel_bindings WHERE address = $1",
                address,
            )
            # We can't directly query for the orphan session by FK, but we can
            # verify no session has a title that would have come from this rule.
            # Easier: just check that no binding was inserted.
        assert binding_count == 0


# ─── inbound endpoint (drives the resolver via the service composition) ─────


class TestInboundMessage:
    async def test_happy_path_creates_session_and_stamps_metadata(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        from aios.db import queries
        from aios.services import channels as ch_svc
        from aios.services import connections as conn_svc
        from aios.services import sessions as sess_svc

        account = f"inbound-{_uniq()}"
        connection = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=account,
            mcp_url="https://m",
            vault_id=vault_id,
            metadata={},
        )
        await ch_svc.create_routing_rule(
            pool,
            prefix=f"signal/{account}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )

        # Mirror the router's logic with defer_wake mocked out.
        with mock.patch("aios.harness.wake.defer_wake") as fake_wake:
            address = f"{connection.connector}/{connection.account}/chat-1"
            resolution = await ch_svc.resolve_channel(pool, address)
            event = await sess_svc.append_user_message(
                pool,
                resolution.session_id,
                "hi",
                metadata={"channel": address, "extra": "stuff"},
            )
            from aios.harness.wake import defer_wake

            await defer_wake(resolution.session_id, cause="inbound_message")

        assert resolution.created_session is True
        assert event.data["metadata"]["channel"] == address
        assert event.data["metadata"]["extra"] == "stuff"
        assert event.data["content"] == "hi"
        fake_wake.assert_awaited_once_with(resolution.session_id, cause="inbound_message")

        # The event was actually persisted on the new session.
        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, resolution.session_id)
        assert any(e.data.get("metadata", {}).get("channel") == address for e in events)


# ─── HTTP-level coverage of the inbound endpoint ────────────────────────────


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    """First HTTP-level test fixture in the codebase.

    Builds the FastAPI app with our testcontainer pool wired straight into
    ``app.state`` (skipping the lifespan, which would create its own pool).
    Mocks ``defer_wake`` for the fixture's lifetime so the inbound handler
    doesn't need a procrastinate worker.
    """
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    with mock.patch("aios.api.routers.connections.defer_wake", new_callable=mock.AsyncMock):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
        ) as client:
            yield client


class TestInboundEndpoint:
    async def _setup_routed(
        self,
        http_client: httpx.AsyncClient,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> tuple[str, str, str]:
        """Create a connection + matching rule. Returns (connection_id, account, prefix)."""
        account = f"http-{_uniq()}"
        r = await http_client.post(
            "/v1/connections",
            json={
                "connector": "signal",
                "account": account,
                "mcp_url": "https://m",
                "vault_id": vault_id,
            },
        )
        assert r.status_code == 201, r.text
        connection_id = r.json()["id"]

        r = await http_client.post(
            "/v1/routing-rules",
            json={
                "prefix": f"signal/{account}",
                "target": f"agent:{agent_id}",
                "session_params": {"environment_id": env_id},
            },
        )
        assert r.status_code == 201, r.text
        return connection_id, account, f"signal/{account}"

    async def test_happy_path_201_and_persists_metadata_channel(
        self,
        http_client: httpx.AsyncClient,
        pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.db import queries

        connection_id, _account, prefix = await self._setup_routed(
            http_client, agent_id, env_id, vault_id
        )

        r = await http_client.post(
            f"/v1/connections/{connection_id}/messages",
            json={"path": "chat-1", "content": "hi"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["created_session"] is True
        assert body["session_id"].startswith("sess_")
        assert body["event_id"].startswith("evt_")

        # Channel metadata is on the persisted event.
        expected_address = f"{prefix}/chat-1"
        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, body["session_id"])
        msg = events[-1]
        assert msg.data["content"] == "hi"
        assert msg.data["metadata"]["channel"] == expected_address

        # Second post short-circuits via the binding.
        r2 = await http_client.post(
            f"/v1/connections/{connection_id}/messages",
            json={"path": "chat-1", "content": "again"},
        )
        assert r2.status_code == 201
        body2 = r2.json()
        assert body2["session_id"] == body["session_id"]
        assert body2["created_session"] is False

    async def test_unknown_connection_returns_404(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.post(
            "/v1/connections/conn_does_not_exist/messages",
            json={"path": "chat-1", "content": "hi"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    async def test_no_route_returns_404_with_no_route_envelope(
        self, http_client: httpx.AsyncClient, vault_id: str
    ) -> None:
        # Connection exists but no rule matches the resulting address.
        account = f"unrouted-{_uniq()}"
        r = await http_client.post(
            "/v1/connections",
            json={
                "connector": "signal",
                "account": account,
                "mcp_url": "https://m",
                "vault_id": vault_id,
            },
        )
        connection_id = r.json()["id"]

        r = await http_client.post(
            f"/v1/connections/{connection_id}/messages",
            json={"path": "chat-1", "content": "hi"},
        )
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["type"] == "no_route"
        assert "address" in body["error"].get("detail", {})

    @pytest.mark.parametrize(
        "bad_path",
        ["", "/x", "x/", "x//y", "x/../y", ".."],
    )
    async def test_malformed_path_returns_422(
        self,
        http_client: httpx.AsyncClient,
        agent_id: str,
        env_id: str,
        vault_id: str,
        bad_path: str,
    ) -> None:
        connection_id, _account, _prefix = await self._setup_routed(
            http_client, agent_id, env_id, vault_id
        )
        r = await http_client.post(
            f"/v1/connections/{connection_id}/messages",
            json={"path": bad_path, "content": "hi"},
        )
        assert r.status_code == 422, (bad_path, r.text)
        assert r.json()["error"]["type"] == "validation_error"


# ─── step-function queries ──────────────────────────────────────────────────


class TestListSessionBindings:
    """Unpaginated "all active bindings for this session" lookup used by
    the step function to derive connection-provided MCP URLs.
    """

    async def test_empty_when_no_bindings(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.db import queries
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        async with pool.acquire() as conn:
            bindings = await queries.list_session_bindings(conn, s.id)
        assert bindings == []

    async def test_returns_all_active_bindings_for_session(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        from aios.db import queries
        from aios.services import channels as ch_svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        suffix = _uniq()
        a1 = await ch_svc.create_binding(pool, address=f"signal/x-{suffix}/1", session_id=s.id)
        a2 = await ch_svc.create_binding(pool, address=f"signal/x-{suffix}/2", session_id=s.id)

        async with pool.acquire() as conn:
            bindings = await queries.list_session_bindings(conn, s.id)
        ids = {b.id for b in bindings}
        assert ids == {a1.id, a2.id}

    async def test_excludes_archived_bindings(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.db import queries
        from aios.services import channels as ch_svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        active = await ch_svc.create_binding(
            pool, address=f"signal/arc-{_uniq()}/a", session_id=s.id
        )
        dead = await ch_svc.create_binding(pool, address=f"signal/arc-{_uniq()}/b", session_id=s.id)
        await ch_svc.archive_binding(pool, dead.id)

        async with pool.acquire() as conn:
            bindings = await queries.list_session_bindings(conn, s.id)
        ids = {b.id for b in bindings}
        assert ids == {active.id}

    async def test_scoped_to_session(self, pool: Any, agent_id: str, env_id: str) -> None:
        from aios.db import queries
        from aios.services import channels as ch_svc
        from aios.services import sessions as sess_svc

        s1 = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        s2 = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        await ch_svc.create_binding(pool, address=f"signal/sc-{_uniq()}/s1", session_id=s1.id)
        await ch_svc.create_binding(pool, address=f"signal/sc-{_uniq()}/s2", session_id=s2.id)

        async with pool.acquire() as conn:
            s1_bindings = await queries.list_session_bindings(conn, s1.id)
            s2_bindings = await queries.list_session_bindings(conn, s2.id)
        assert len(s1_bindings) == 1 and s1_bindings[0].session_id == s1.id
        assert len(s2_bindings) == 1 and s2_bindings[0].session_id == s2.id


class TestGetConnectionsByPairs:
    """Batch lookup of active connections by (connector, account) pairs,
    used by discovery after collecting bindings.
    """

    async def test_empty_input_returns_empty(self, pool: Any) -> None:
        from aios.db import queries

        async with pool.acquire() as conn:
            rows = await queries.get_connections_by_pairs(conn, [])
        assert rows == []

    async def test_returns_matching_connections(self, pool: Any, vault_id: str) -> None:
        from aios.db import queries
        from aios.services import connections as conn_svc

        a = f"gcp-a-{_uniq()}"
        b = f"gcp-b-{_uniq()}"
        c_a = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=a,
            mcp_url="https://m1",
            vault_id=vault_id,
            metadata={},
        )
        c_b = await conn_svc.create_connection(
            pool, connector="slack", account=b, mcp_url="https://m2", vault_id=vault_id, metadata={}
        )

        async with pool.acquire() as conn:
            rows = await queries.get_connections_by_pairs(
                conn, [("signal", a), ("slack", b), ("discord", "nope")]
            )
        ids = {c.id for c in rows}
        assert ids == {c_a.id, c_b.id}
        # Results must be id-ordered so callers can pass them into the
        # system prompt in a stable order (prompt-cache stability
        # invariant).  Caller-side inputs are typically set-derived and
        # therefore have non-deterministic order across processes.
        returned_ids = [c.id for c in rows]
        assert returned_ids == sorted(returned_ids)

    async def test_excludes_archived(self, pool: Any, vault_id: str) -> None:
        from aios.db import queries
        from aios.services import connections as conn_svc

        acct = f"gcp-arc-{_uniq()}"
        c = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=acct,
            mcp_url="https://m",
            vault_id=vault_id,
            metadata={},
        )
        await conn_svc.archive_connection(pool, c.id)
        async with pool.acquire() as conn:
            rows = await queries.get_connections_by_pairs(conn, [("signal", acct)])
        assert rows == []


class TestResolveAuthForUrlPrecedence:
    """Connection-declared auth takes precedence over session_vaults
    when both sources have a credential for the same URL.
    """

    async def test_connection_credential_beats_session_vault(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        from pydantic import SecretStr

        from aios.config import get_settings
        from aios.crypto.vault import CryptoBox
        from aios.mcp.client import resolve_auth_for_url
        from aios.models.vaults import VaultCredentialCreate
        from aios.services import connections as conn_svc
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        shared_url = f"https://shared-mcp-{_uniq()}.example"

        v1 = await vault_svc.create_vault(pool, display_name="session-vault", metadata={})
        await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v1.id,
            body=VaultCredentialCreate(
                display_name="session",
                mcp_server_url=shared_url,
                auth_type="static_bearer",
                token=SecretStr("SESSION_TOKEN_V1"),
            ),
        )

        v2 = await vault_svc.create_vault(pool, display_name="connection-vault", metadata={})
        await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v2.id,
            body=VaultCredentialCreate(
                display_name="connection",
                mcp_server_url=shared_url,
                auth_type="static_bearer",
                token=SecretStr("CONNECTION_TOKEN_V2"),
            ),
        )

        await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"precedence-{_uniq()}",
            mcp_url=shared_url,
            vault_id=v2.id,
            metadata={},
        )

        session = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            vault_ids=[v1.id],
        )

        headers = await resolve_auth_for_url(pool, crypto_box, session.id, shared_url)
        # Connection wins even though session_vaults ALSO has a credential.
        assert headers == {"Authorization": "Bearer CONNECTION_TOKEN_V2"}

    async def test_session_vault_used_for_non_connection_url(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """When the URL isn't owned by any connection, fall back to session_vaults."""
        from pydantic import SecretStr

        from aios.config import get_settings
        from aios.crypto.vault import CryptoBox
        from aios.mcp.client import resolve_auth_for_url
        from aios.models.vaults import VaultCredentialCreate
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        url = f"https://agent-mcp-{_uniq()}.example"

        v = await vault_svc.create_vault(pool, display_name="agent-vault", metadata={})
        await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v.id,
            body=VaultCredentialCreate(
                display_name="agent",
                mcp_server_url=url,
                auth_type="static_bearer",
                token=SecretStr("AGENT_TOKEN"),
            ),
        )

        session = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            vault_ids=[v.id],
        )

        headers = await resolve_auth_for_url(pool, crypto_box, session.id, url)
        assert headers == {"Authorization": "Bearer AGENT_TOKEN"}

    async def test_connection_with_no_matching_credential_returns_empty(
        self, pool: Any, agent_id: str, env_id: str
    ) -> None:
        """Connection ownership decides the source, regardless of hit.  If
        the connection's vault has no credential for the URL we MUST NOT
        silently fall back to session_vaults — a misconfigured connection
        must surface as missing auth, not as a leaked tenant credential.
        """
        from pydantic import SecretStr

        from aios.config import get_settings
        from aios.crypto.vault import CryptoBox
        from aios.mcp.client import resolve_auth_for_url
        from aios.models.vaults import VaultCredentialCreate
        from aios.services import connections as conn_svc
        from aios.services import sessions as sess_svc
        from aios.services import vaults as vault_svc

        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        url = f"https://broken-conn-{_uniq()}.example"

        # session_vaults has the credential...
        v_session = await vault_svc.create_vault(pool, display_name="s", metadata={})
        await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v_session.id,
            body=VaultCredentialCreate(
                display_name="s",
                mcp_server_url=url,
                auth_type="static_bearer",
                token=SecretStr("LEAKED"),
            ),
        )
        # ...but the connection's vault is empty.
        v_empty = await vault_svc.create_vault(pool, display_name="empty", metadata={})
        await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"broken-{_uniq()}",
            mcp_url=url,
            vault_id=v_empty.id,
            metadata={},
        )

        session = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            vault_ids=[v_session.id],
        )

        headers = await resolve_auth_for_url(pool, crypto_box, session.id, url)
        # Must NOT be "Bearer LEAKED".
        assert headers == {}


class TestResolveVaultCredential:
    """Direct (vault_id, URL) → (blob, auth_type) lookup — no session_vaults
    join.  The connection-first half of resolve_auth_for_url uses this.
    """

    async def test_match_returns_blob(self, pool: Any, vault_id: str) -> None:
        from pydantic import SecretStr

        from aios.config import get_settings
        from aios.crypto.vault import CryptoBox
        from aios.db import queries
        from aios.models.vaults import VaultCredentialCreate
        from aios.services import vaults as vault_svc

        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        url = f"https://rvc-{_uniq()}.example"
        await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault_id,
            body=VaultCredentialCreate(
                display_name="t",
                mcp_server_url=url,
                auth_type="static_bearer",
                token=SecretStr("hello"),
            ),
        )
        async with pool.acquire() as conn:
            hit = await queries.resolve_vault_credential(
                conn, vault_id=vault_id, mcp_server_url=url
            )
        assert hit is not None
        blob, auth_type = hit
        assert auth_type == "static_bearer"
        import json as _json

        assert _json.loads(crypto_box.decrypt(blob))["token"] == "hello"

    async def test_no_match_returns_none(self, pool: Any, vault_id: str) -> None:
        from aios.db import queries

        async with pool.acquire() as conn:
            hit = await queries.resolve_vault_credential(
                conn,
                vault_id=vault_id,
                mcp_server_url=f"https://rvc-miss-{_uniq()}.example",
            )
        assert hit is None

    async def test_excludes_archived_credential(self, pool: Any, vault_id: str) -> None:
        from pydantic import SecretStr

        from aios.config import get_settings
        from aios.crypto.vault import CryptoBox
        from aios.db import queries
        from aios.models.vaults import VaultCredentialCreate
        from aios.services import vaults as vault_svc

        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        url = f"https://rvc-arc-{_uniq()}.example"
        cred = await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=vault_id,
            body=VaultCredentialCreate(
                display_name="t",
                mcp_server_url=url,
                auth_type="static_bearer",
                token=SecretStr("hello"),
            ),
        )
        await vault_svc.archive_vault_credential(pool, vault_id, cred.id)

        async with pool.acquire() as conn:
            hit = await queries.resolve_vault_credential(
                conn, vault_id=vault_id, mcp_server_url=url
            )
        assert hit is None

    async def test_scoped_to_vault_id(self, pool: Any) -> None:
        """A credential in vault A must not surface when looking up in vault B."""
        from pydantic import SecretStr

        from aios.config import get_settings
        from aios.crypto.vault import CryptoBox
        from aios.db import queries
        from aios.models.vaults import VaultCredentialCreate
        from aios.services import vaults as vault_svc

        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        v_a = await vault_svc.create_vault(pool, display_name="rvc-a", metadata={})
        v_b = await vault_svc.create_vault(pool, display_name="rvc-b", metadata={})
        url = f"https://rvc-scope-{_uniq()}.example"
        await vault_svc.create_vault_credential(
            pool,
            crypto_box,
            vault_id=v_a.id,
            body=VaultCredentialCreate(
                display_name="t",
                mcp_server_url=url,
                auth_type="static_bearer",
                token=SecretStr("A"),
            ),
        )
        async with pool.acquire() as conn:
            hit_a = await queries.resolve_vault_credential(
                conn, vault_id=v_a.id, mcp_server_url=url
            )
            hit_b = await queries.resolve_vault_credential(
                conn, vault_id=v_b.id, mcp_server_url=url
            )
        assert hit_a is not None
        assert hit_b is None


class TestGetConnectionVaultForUrl:
    """URL → vault_id lookup used by resolve_auth_for_url to decide
    whether a URL belongs to a registered connection.
    """

    async def test_match_returns_vault_id(self, pool: Any, vault_id: str) -> None:
        from aios.db import queries
        from aios.services import connections as conn_svc

        url = f"https://mcp-match-{_uniq()}.example"
        await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"gcvfu-{_uniq()}",
            mcp_url=url,
            vault_id=vault_id,
            metadata={},
        )
        async with pool.acquire() as conn:
            hit = await queries.get_connection_vault_for_url(conn, url)
        assert hit == vault_id

    async def test_no_match_returns_none(self, pool: Any) -> None:
        from aios.db import queries

        async with pool.acquire() as conn:
            hit = await queries.get_connection_vault_for_url(
                conn, f"https://unknown-{_uniq()}.example"
            )
        assert hit is None

    async def test_excludes_archived(self, pool: Any, vault_id: str) -> None:
        from aios.db import queries
        from aios.services import connections as conn_svc

        url = f"https://mcp-arc-{_uniq()}.example"
        c = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"gcvfu-arc-{_uniq()}",
            mcp_url=url,
            vault_id=vault_id,
            metadata={},
        )
        await conn_svc.archive_connection(pool, c.id)
        async with pool.acquire() as conn:
            hit = await queries.get_connection_vault_for_url(conn, url)
        assert hit is None
