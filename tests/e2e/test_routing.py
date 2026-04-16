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
