"""E2E tests for the routing infrastructure (issue #30).

Covers connection / channel binding / routing rule CRUD plus the
``resolve_channel`` resolver and the nested ``POST /v1/connections/
{id}/routing-rules`` + inbound-message endpoints.  Runs against a real
testcontainer Postgres with migrations applied.
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
    """Short hex unique across the whole test session."""
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


@pytest.fixture
async def connection(pool: Any, vault_id: str) -> Any:
    """Fresh signal connection per test.  Rules are scoped to this
    connection's id; addresses constructed against ``connection.connector``
    + ``connection.account``.
    """
    from aios.services import connections as svc

    return await svc.create_connection(
        pool,
        connector="signal",
        account=f"t-{_uniq()}",
        mcp_url="https://mcp.example.com",
        vault_id=vault_id,
        metadata={},
    )


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
        # archived connection no longer counts towards uniqueness
        await svc.create_connection(
            pool,
            connector="signal",
            account=account,
            mcp_url="https://m2",
            vault_id=vault_id,
            metadata={},
        )

    async def test_archive_blocked_by_active_bindings(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """A connection with an active binding can't be archived — doing so
        would drop MCP tools from any live session bound to channels under
        that connection.
        """
        from aios.services import channels as ch_svc
        from aios.services import connections as svc
        from aios.services import sessions as sess_svc

        session = await sess_svc.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
        )
        binding = await ch_svc.create_binding(
            pool,
            address=f"{connection.connector}/{connection.account}/chat-1",
            session_id=session.id,
        )

        with pytest.raises(ConflictError) as exc:
            await svc.archive_connection(pool, connection.id)
        assert "active channel binding" in str(exc.value)
        assert exc.value.detail["active_bindings"] == 1

        # After archiving the binding, connection archival succeeds.
        await ch_svc.archive_binding(pool, binding.id)
        archived = await svc.archive_connection(pool, connection.id)
        assert archived.archived_at is not None

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
    async def test_create_agent_target(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.services import channels as svc

        r = await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="chat-a",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        assert r.connection_id == connection.id
        assert r.prefix == "chat-a"
        assert r.target == f"agent:{agent_id}"

    async def test_create_empty_prefix_catch_all(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Empty prefix is the per-connection catch-all — accepted."""
        from aios.services import channels as svc

        r = await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        assert r.prefix == ""

    async def test_create_session_target_rejects_session_params(
        self, pool: Any, connection: Any
    ) -> None:
        from aios.services import channels as svc

        with pytest.raises(ValidationError):
            await svc.create_routing_rule(
                pool,
                connection.id,
                prefix="bad",
                target="session:sess_xxx",
                session_params=SessionParams(environment_id="env_x"),
            )

    async def test_agent_target_requires_environment(
        self, pool: Any, agent_id: str, connection: Any
    ) -> None:
        from aios.services import channels as svc

        with pytest.raises(ValidationError):
            await svc.create_routing_rule(
                pool,
                connection.id,
                prefix="missing-env",
                target=f"agent:{agent_id}",
                session_params=SessionParams(),
            )

    async def test_unique_per_connection_prefix(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Same prefix on the same connection is a conflict."""
        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="uniq",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        with pytest.raises(ConflictError):
            await svc.create_routing_rule(
                pool,
                connection.id,
                prefix="uniq",
                target=f"agent:{agent_id}",
                session_params=SessionParams(environment_id=env_id),
            )

    async def test_same_prefix_on_different_connections_allowed(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        """Uniqueness is ``(connection_id, prefix)`` — not global."""
        from aios.services import channels as svc
        from aios.services import connections as conn_svc

        c1 = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"crossA-{_uniq()}",
            mcp_url="https://a",
            vault_id=vault_id,
            metadata={},
        )
        c2 = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"crossB-{_uniq()}",
            mcp_url="https://b",
            vault_id=vault_id,
            metadata={},
        )
        params = SessionParams(environment_id=env_id)
        await svc.create_routing_rule(
            pool, c1.id, prefix="shared", target=f"agent:{agent_id}", session_params=params
        )
        # Same prefix on a different connection is fine.
        await svc.create_routing_rule(
            pool, c2.id, prefix="shared", target=f"agent:{agent_id}", session_params=params
        )

    async def test_update_target(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.services import channels as svc

        r = await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="upd",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        updated = await svc.update_routing_rule(
            pool, connection.id, r.id, target=f"agent:{agent_id}@1"
        )
        assert updated.target == f"agent:{agent_id}@1"

    async def test_archive(self, pool: Any, agent_id: str, env_id: str, connection: Any) -> None:
        from aios.services import channels as svc

        r = await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="arch",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        archived = await svc.archive_routing_rule(pool, connection.id, r.id)
        assert archived.archived_at is not None

    async def test_wrong_connection_scope_404s(
        self, pool: Any, agent_id: str, env_id: str, connection: Any, vault_id: str
    ) -> None:
        """A rule on connection A isn't visible through connection B's scope."""
        from aios.services import channels as svc
        from aios.services import connections as conn_svc

        r = await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="scoped",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        other = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"other-{_uniq()}",
            mcp_url="https://x",
            vault_id=vault_id,
            metadata={},
        )
        with pytest.raises(NotFoundError):
            await svc.get_routing_rule(pool, other.id, r.id)


# ─── find_matching_rule ─────────────────────────────────────────────────────


class TestFindMatchingRule:
    async def test_exact_match(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.db import queries
        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="exact",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, connection.id, "exact")
        assert r is not None
        assert r.prefix == "exact"

    async def test_empty_prefix_catch_all(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Empty prefix matches any path when no specific rule wins."""
        from aios.db import queries
        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, connection.id, "anything/at/all")
        assert r is not None
        assert r.prefix == ""

    async def test_specific_beats_catch_all(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Longest-prefix sort: a concrete match wins over the empty catch-all."""
        from aios.db import queries
        from aios.services import channels as svc

        params = SessionParams(environment_id=env_id)
        await svc.create_routing_rule(
            pool, connection.id, prefix="", target=f"agent:{agent_id}", session_params=params
        )
        await svc.create_routing_rule(
            pool, connection.id, prefix="group", target=f"agent:{agent_id}", session_params=params
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, connection.id, "group/sub")
        assert r is not None
        assert r.prefix == "group"

    async def test_longer_prefix_wins(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.db import queries
        from aios.services import channels as svc

        params = SessionParams(environment_id=env_id)
        await svc.create_routing_rule(
            pool, connection.id, prefix="grp", target=f"agent:{agent_id}", session_params=params
        )
        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="grp/specific",
            target=f"agent:{agent_id}",
            session_params=params,
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, connection.id, "grp/specific/x")
        assert r is not None
        assert r.prefix == "grp/specific"

    async def test_segment_aware(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """``foo`` must NOT match ``foofoo`` — segment boundary required."""
        from aios.db import queries
        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="seg",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            none_match = await queries.find_matching_rule(conn, connection.id, "segfoo")
            yes_match = await queries.find_matching_rule(conn, connection.id, "seg/abc")
        assert none_match is None
        assert yes_match is not None

    async def test_cross_connection_isolation(
        self, pool: Any, agent_id: str, env_id: str, connection: Any, vault_id: str
    ) -> None:
        """A rule on connection A must not match a path queried with connection B's id."""
        from aios.db import queries
        from aios.services import channels as svc
        from aios.services import connections as conn_svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="isolated",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        other = await conn_svc.create_connection(
            pool,
            connector="signal",
            account=f"other-{_uniq()}",
            mcp_url="https://x",
            vault_id=vault_id,
            metadata={},
        )
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, other.id, "isolated/x")
        assert r is None

    async def test_no_match_returns_none(self, pool: Any, connection: Any) -> None:
        from aios.db import queries

        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, connection.id, "nope/x")
        assert r is None

    async def test_archived_rule_excluded(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.db import queries
        from aios.services import channels as svc

        rule = await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="arch-rule",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        await svc.archive_routing_rule(pool, connection.id, rule.id)
        async with pool.acquire() as conn:
            r = await queries.find_matching_rule(conn, connection.id, "arch-rule/x")
        assert r is None

    async def test_underscore_in_prefix_treated_literally(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """LIKE meta-characters in prefix must NOT act as wildcards."""
        from aios.db import queries
        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="fo_bar",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        async with pool.acquire() as conn:
            wrong = await queries.find_matching_rule(conn, connection.id, "fooXbar/x")
            right = await queries.find_matching_rule(conn, connection.id, "fo_bar/x")
        assert wrong is None
        assert right is not None


# ─── resolve_channel ────────────────────────────────────────────────────────


class TestResolveChannel:
    async def test_existing_binding_short_circuits(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        address = f"{connection.connector}/{connection.account}/chat-x"
        binding = await svc.create_binding(pool, address=address, session_id=s.id)

        result = await svc.resolve_channel(pool, connection, "chat-x")
        assert result.session_id == s.id
        assert result.binding_id == binding.id
        assert result.created_session is False

    async def test_session_target(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="st",
            target=f"session:{s.id}",
            session_params=SessionParams(),
        )
        result = await svc.resolve_channel(pool, connection, "st/whatever")
        assert result.session_id == s.id
        assert result.created_session is False

    async def test_agent_target_creates_session(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str, connection: Any
    ) -> None:
        from aios.db import queries
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="at",
            target=f"agent:{agent_id}",
            session_params=SessionParams(
                environment_id=env_id,
                vault_ids=[vault_id],
                title="Routed: {address}",
                metadata={"source": "rule"},
            ),
        )
        result = await svc.resolve_channel(pool, connection, "at/chat-1")
        assert result.created_session is True

        address = f"{connection.connector}/{connection.account}/at/chat-1"
        s = await sess_svc.get_session(pool, result.session_id)
        assert s.title == f"Routed: {address}"
        assert s.metadata == {"source": "rule"}
        async with pool.acquire() as conn:
            vids = await queries.get_session_vault_ids(conn, s.id)
        assert vids == [vault_id]

        again = await svc.resolve_channel(pool, connection, "at/chat-1")
        assert again.session_id == result.session_id
        assert again.binding_id == result.binding_id
        assert again.created_session is False

    async def test_no_route_raises_when_connection_has_no_rules(
        self, pool: Any, connection: Any
    ) -> None:
        from aios.services import channels as svc

        with pytest.raises(NoRouteError):
            await svc.resolve_channel(pool, connection, "unrouted")

    async def test_concurrent_resolve_same_address_returns_same_session(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Two simultaneous first-time resolves on the same address must
        agree on the session — no spurious conflict, no orphan session.
        """
        import asyncio

        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="race",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )

        results = await asyncio.gather(
            svc.resolve_channel(pool, connection, "race/chat-1"),
            svc.resolve_channel(pool, connection, "race/chat-1"),
        )
        assert results[0].session_id == results[1].session_id
        assert results[0].binding_id == results[1].binding_id
        assert sum(r.created_session for r in results) == 1

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM channel_bindings "
                "WHERE connection_id = $1 AND path = $2 AND archived_at IS NULL",
                connection.id,
                "race/chat-1",
            )
        assert count == 1

    async def test_concurrent_resolve_different_paths_do_not_block(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Advisory lock is per-address — distinct paths resolve in parallel."""
        import asyncio

        from aios.services import channels as svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="par",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        results = await asyncio.gather(
            svc.resolve_channel(pool, connection, "par/a"),
            svc.resolve_channel(pool, connection, "par/b"),
        )
        assert results[0].session_id != results[1].session_id
        assert all(r.created_session for r in results)

    async def test_falls_through_archived_binding(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Archived binding must NOT short-circuit; resolver falls to rules."""
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="arch-bind",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        old_session = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        address = f"{connection.connector}/{connection.account}/arch-bind/x"
        old_binding = await svc.create_binding(pool, address=address, session_id=old_session.id)
        await svc.archive_binding(pool, old_binding.id)

        result = await svc.resolve_channel(pool, connection, "arch-bind/x")
        assert result.session_id != old_session.id
        assert result.created_session is True

    async def test_session_target_with_archived_session_raises(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.services import channels as svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        await sess_svc.archive_session(pool, s.id)
        await svc.create_routing_rule(
            pool,
            connection.id,
            prefix="st-arch",
            target=f"session:{s.id}",
            session_params=SessionParams(),
        )
        with pytest.raises(NotFoundError, match="archived"):
            await svc.resolve_channel(pool, connection, "st-arch/x")


# ─── inbound endpoint (full HTTP) ───────────────────────────────────────────


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    """FastAPI test client wired to the testcontainer pool."""
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


class TestNestedRoutingRulesEndpoint:
    async def test_create_and_get_nested(
        self, http_client: httpx.AsyncClient, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        r = await http_client.post(
            "/v1/connections",
            json={
                "connector": "signal",
                "account": f"nested-{_uniq()}",
                "mcp_url": "https://m",
                "vault_id": vault_id,
            },
        )
        assert r.status_code == 201, r.text
        connection_id = r.json()["id"]

        r = await http_client.post(
            f"/v1/connections/{connection_id}/routing-rules",
            json={
                "prefix": "chat-a",
                "target": f"agent:{agent_id}",
                "session_params": {"environment_id": env_id},
            },
        )
        assert r.status_code == 201, r.text
        rule = r.json()
        assert rule["connection_id"] == connection_id
        assert rule["prefix"] == "chat-a"

        r = await http_client.get(f"/v1/connections/{connection_id}/routing-rules/{rule['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == rule["id"]

    async def test_unknown_connection_404s(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.get("/v1/connections/conn_does_not_exist/routing-rules")
        assert r.status_code == 404

    async def test_rule_scoped_to_connection(
        self, http_client: httpx.AsyncClient, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        """A rule on connection A is invisible through connection B."""
        a = await http_client.post(
            "/v1/connections",
            json={
                "connector": "signal",
                "account": f"scopeA-{_uniq()}",
                "mcp_url": "https://a",
                "vault_id": vault_id,
            },
        )
        b = await http_client.post(
            "/v1/connections",
            json={
                "connector": "signal",
                "account": f"scopeB-{_uniq()}",
                "mcp_url": "https://b",
                "vault_id": vault_id,
            },
        )
        aid = a.json()["id"]
        bid = b.json()["id"]
        rr = await http_client.post(
            f"/v1/connections/{aid}/routing-rules",
            json={
                "prefix": "x",
                "target": f"agent:{agent_id}",
                "session_params": {"environment_id": env_id},
            },
        )
        rule_id = rr.json()["id"]

        # Under connection B, the rule doesn't exist.
        r = await http_client.get(f"/v1/connections/{bid}/routing-rules/{rule_id}")
        assert r.status_code == 404


class TestInboundEndpoint:
    async def _setup_routed(
        self,
        http_client: httpx.AsyncClient,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> tuple[str, str]:
        """Create a connection + catch-all rule.  Returns (connection_id, account)."""
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
            f"/v1/connections/{connection_id}/routing-rules",
            json={
                "prefix": "",
                "target": f"agent:{agent_id}",
                "session_params": {"environment_id": env_id},
            },
        )
        assert r.status_code == 201, r.text
        return connection_id, account

    async def test_happy_path_201_and_persists_metadata_channel(
        self,
        http_client: httpx.AsyncClient,
        pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.db import queries

        connection_id, account = await self._setup_routed(http_client, agent_id, env_id, vault_id)

        r = await http_client.post(
            f"/v1/connections/{connection_id}/messages",
            json={"path": "chat-1", "content": "hi"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["created_session"] is True
        assert body["session_id"].startswith("sess_")

        expected_address = f"signal/{account}/chat-1"
        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, body["session_id"])
        msg = events[-1]
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

    async def test_no_route_returns_404(
        self, http_client: httpx.AsyncClient, vault_id: str
    ) -> None:
        """Connection exists but no rule matches the resulting path."""
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
        assert r.json()["error"]["type"] == "no_route"

    @pytest.mark.parametrize("bad_path", ["", "/x", "x/", "x//y", "x/../y", ".."])
    async def test_malformed_path_returns_422(
        self,
        http_client: httpx.AsyncClient,
        agent_id: str,
        env_id: str,
        vault_id: str,
        bad_path: str,
    ) -> None:
        connection_id, _ = await self._setup_routed(http_client, agent_id, env_id, vault_id)
        r = await http_client.post(
            f"/v1/connections/{connection_id}/messages",
            json={"path": bad_path, "content": "hi"},
        )
        assert r.status_code == 422, (bad_path, r.text)


# ─── step-function queries ──────────────────────────────────────────────────


class TestListSessionBindings:
    """Unpaginated "all bindings for this session" lookup used by the
    step function to derive connection-provided MCP URLs.
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

    async def test_returns_all_active_bindings_with_address(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        """Listing returns bindings with the display address reconstructed."""
        from aios.db import queries
        from aios.services import channels as ch_svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        base = f"{connection.connector}/{connection.account}"
        a1 = await ch_svc.create_binding(pool, address=f"{base}/chat-1", session_id=s.id)
        a2 = await ch_svc.create_binding(pool, address=f"{base}/chat-2", session_id=s.id)

        async with pool.acquire() as conn:
            bindings = await queries.list_session_bindings(conn, s.id)
        ids = {b.id for b in bindings}
        assert ids == {a1.id, a2.id}
        addresses = {b.address for b in bindings}
        assert addresses == {f"{base}/chat-1", f"{base}/chat-2"}

    async def test_excludes_archived_bindings(
        self, pool: Any, agent_id: str, env_id: str, connection: Any
    ) -> None:
        from aios.db import queries
        from aios.services import channels as ch_svc
        from aios.services import sessions as sess_svc

        s = await sess_svc.create_session(
            pool, agent_id=agent_id, environment_id=env_id, title=None, metadata={}
        )
        base = f"{connection.connector}/{connection.account}"
        active = await ch_svc.create_binding(pool, address=f"{base}/a", session_id=s.id)
        dead = await ch_svc.create_binding(pool, address=f"{base}/b", session_id=s.id)
        await ch_svc.archive_binding(pool, dead.id)

        async with pool.acquire() as conn:
            bindings = await queries.list_session_bindings(conn, s.id)
        assert {b.id for b in bindings} == {active.id}


class TestGetConnectionsByPairs:
    """Batch lookup of connections by (connector, account) pairs."""

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
            pool,
            connector="slack",
            account=b,
            mcp_url="https://m2",
            vault_id=vault_id,
            metadata={},
        )

        async with pool.acquire() as conn:
            rows = await queries.get_connections_by_pairs(
                conn, [("signal", a), ("slack", b), ("discord", "nope")]
            )
        ids = {c.id for c in rows}
        assert ids == {c_a.id, c_b.id}
        # Stable id-order for prompt-cache stability.
        returned_ids = [c.id for c in rows]
        assert returned_ids == sorted(returned_ids)
