"""Unit tests for MCP client — discover, call, auth resolution.

All MCP SDK interactions are mocked. No network calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.crypto.vault import CryptoBox
from aios.mcp.client import call_mcp_tool, discover_mcp_tools, resolve_auth_for_url
from tests.unit.conftest import fake_pool_yielding_conn

# ── resolve_auth_for_url ──────────────────────────────────────────────────────


class TestResolveAuthForUrl:
    """Connection-declared credentials take precedence over session_vaults,
    and mcp_oauth credentials are transparently refreshed when expiring.
    """

    @pytest.fixture
    def crypto_box(self) -> CryptoBox:
        import os

        return CryptoBox(os.urandom(32))

    # ── precedence: connection-owned URLs ─────────────────────────────────

    async def test_connection_url_uses_connection_vault(self, crypto_box: CryptoBox) -> None:
        payload = json.dumps({"token": "connection-token"})
        blob = crypto_box.encrypt(payload)
        pool = fake_pool_yielding_conn(MagicMock())
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_vault_credential",
                new_callable=AsyncMock,
            ) as v,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
        ):
            g.return_value = "vlt_v2"
            v.return_value = (blob, "static_bearer")
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )
        assert result == {"Authorization": "Bearer connection-token"}
        # Connection precedence: session_vaults lookup MUST NOT be consulted.
        s.assert_not_awaited()
        v.assert_awaited_once()

    async def test_connection_url_missing_credential_returns_empty(
        self, crypto_box: CryptoBox
    ) -> None:
        pool = fake_pool_yielding_conn(MagicMock())
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_vault_credential",
                new_callable=AsyncMock,
            ) as v,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
        ):
            g.return_value = "vlt_v2"
            v.return_value = None
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )
        assert result == {}
        # Still doesn't fall back — connection ownership decided the source.
        s.assert_not_awaited()

    async def test_mcp_oauth_from_connection(self, crypto_box: CryptoBox) -> None:
        payload = json.dumps({"access_token": "oauth-conn-token"})
        blob = crypto_box.encrypt(payload)
        pool = fake_pool_yielding_conn(MagicMock())
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_vault_credential",
                new_callable=AsyncMock,
            ) as v,
        ):
            g.return_value = "vlt_v2"
            v.return_value = (blob, "mcp_oauth")
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )
        assert result == {"Authorization": "Bearer oauth-conn-token"}

    # ── fallback: session_vaults ──────────────────────────────────────────

    async def test_non_connection_url_falls_back_to_session_vaults(
        self, crypto_box: CryptoBox
    ) -> None:
        payload = json.dumps({"token": "session-token"})
        blob = crypto_box.encrypt(payload)
        pool = fake_pool_yielding_conn(MagicMock())
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_vault_credential",
                new_callable=AsyncMock,
            ) as v,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
        ):
            g.return_value = None
            s.return_value = (blob, "static_bearer", "vlt_s1")
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )
        assert result == {"Authorization": "Bearer session-token"}
        v.assert_not_awaited()
        s.assert_awaited_once()

    async def test_neither_source_returns_empty(self, crypto_box: CryptoBox) -> None:
        pool = fake_pool_yielding_conn(MagicMock())
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
        ):
            g.return_value = None
            s.return_value = None
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )
        assert result == {}

    async def test_empty_token_returns_empty(self, crypto_box: CryptoBox) -> None:
        payload = json.dumps({"token": ""})
        blob = crypto_box.encrypt(payload)
        pool = fake_pool_yielding_conn(MagicMock())
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
        ):
            g.return_value = None
            s.return_value = (blob, "static_bearer", "vlt_s1")
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )
        assert result == {}

    # ── OAuth refresh (applies to both paths) ─────────────────────────────

    async def test_oauth_refresh_triggered_when_expiring(self, crypto_box: CryptoBox) -> None:
        """Session-vaults path: stale mcp_oauth token triggers refresh, then
        the re-read returns the fresh token."""
        from datetime import UTC, datetime, timedelta

        expiring = json.dumps(
            {
                "access_token": "stale",
                "expires_at": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
            }
        )
        fresh = json.dumps({"access_token": "fresh"})
        stale_blob = crypto_box.encrypt(expiring)
        fresh_blob = crypto_box.encrypt(fresh)
        pool = fake_pool_yielding_conn(MagicMock())

        refresh_mock = AsyncMock()
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
            patch(
                "aios.mcp.client.queries.resolve_vault_credential",
                new_callable=AsyncMock,
            ) as v,
            patch("aios.mcp.client.refresh_credential", refresh_mock),
        ):
            g.return_value = None
            s.return_value = (stale_blob, "mcp_oauth", "vlt_s1")
            v.return_value = (fresh_blob, "mcp_oauth")
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )

        refresh_mock.assert_awaited_once()
        kwargs = refresh_mock.await_args.kwargs
        assert kwargs["vault_id"] == "vlt_s1"
        assert kwargs["mcp_server_url"] == "https://mcp.example.com"
        assert result == {"Authorization": "Bearer fresh"}

    async def test_oauth_refresh_triggered_on_connection_path(self, crypto_box: CryptoBox) -> None:
        """Connection-owned path: refresh ALSO triggers here — the refresh
        flow is source-agnostic, it just needs a vault_id + url pair.
        """
        from datetime import UTC, datetime, timedelta

        expiring = json.dumps(
            {
                "access_token": "stale-conn",
                "expires_at": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
            }
        )
        fresh = json.dumps({"access_token": "fresh-conn"})
        stale_blob = crypto_box.encrypt(expiring)
        fresh_blob = crypto_box.encrypt(fresh)
        pool = fake_pool_yielding_conn(MagicMock())

        refresh_mock = AsyncMock()
        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_vault_credential",
                new_callable=AsyncMock,
            ) as v,
            patch("aios.mcp.client.refresh_credential", refresh_mock),
        ):
            g.return_value = "vlt_conn"
            # Two reads: initial (stale) and post-refresh (fresh).
            v.side_effect = [(stale_blob, "mcp_oauth"), (fresh_blob, "mcp_oauth")]
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )

        refresh_mock.assert_awaited_once()
        assert refresh_mock.await_args.kwargs["vault_id"] == "vlt_conn"
        assert result == {"Authorization": "Bearer fresh-conn"}

    async def test_oauth_refresh_not_triggered_when_fresh(self, crypto_box: CryptoBox) -> None:
        from datetime import UTC, datetime, timedelta

        far_future = json.dumps(
            {
                "access_token": "still-good",
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }
        )
        blob = crypto_box.encrypt(far_future)
        pool = fake_pool_yielding_conn(MagicMock())
        refresh_mock = AsyncMock()

        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
            patch("aios.mcp.client.refresh_credential", refresh_mock),
        ):
            g.return_value = None
            s.return_value = (blob, "mcp_oauth", "vlt_s1")
            result = await resolve_auth_for_url(
                pool, crypto_box, "sess_123", "https://mcp.example.com"
            )

        refresh_mock.assert_not_awaited()
        assert result == {"Authorization": "Bearer still-good"}

    async def test_oauth_refresh_failure_bubbles(self, crypto_box: CryptoBox) -> None:
        from datetime import UTC, datetime, timedelta

        from aios.errors import OAuthRefreshError

        expiring = json.dumps(
            {
                "access_token": "stale",
                "expires_at": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
            }
        )
        blob = crypto_box.encrypt(expiring)
        pool = fake_pool_yielding_conn(MagicMock())

        with (
            patch(
                "aios.mcp.client.queries.get_connection_vault_for_url",
                new_callable=AsyncMock,
            ) as g,
            patch(
                "aios.mcp.client.queries.resolve_mcp_credential",
                new_callable=AsyncMock,
            ) as s,
            patch(
                "aios.mcp.client.refresh_credential",
                AsyncMock(side_effect=OAuthRefreshError("bad", detail={})),
            ),
            pytest.raises(OAuthRefreshError),
        ):
            g.return_value = None
            s.return_value = (blob, "mcp_oauth", "vlt_s1")
            await resolve_auth_for_url(pool, crypto_box, "sess_123", "https://mcp.example.com")


# ── discover_mcp_tools ────────────────────────────────────────────────────────


def _make_mock_tool(name: str, description: str, schema: dict[str, Any]) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema
    return tool


def _mock_init_result(instructions: str | None = None) -> MagicMock:
    """Build an ``InitializeResult``-shaped mock.

    ``MagicMock`` would happily synthesise a sub-mock for ``.instructions``
    (truthy by default), so tests must set the attribute explicitly to
    cover the ``None`` path.
    """
    result = MagicMock()
    result.instructions = instructions
    return result


class TestDiscoverMcpTools:
    async def test_discovery_returns_namespaced_tools(self) -> None:
        mock_tool = _make_mock_tool("create_issue", "Create a GitHub issue", {"type": "object"})
        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=_mock_init_result())
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        with (
            patch("aios.mcp.client.streamable_http_client") as mock_transport,
            patch("aios.mcp.client.ClientSession") as mock_session_cls,
        ):
            mock_transport.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tools, instructions = await discover_mcp_tools("https://mcp.github.com/", "github", {})

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "mcp__github__create_issue"
        assert tools[0]["function"]["description"] == "Create a GitHub issue"
        assert tools[0]["function"]["parameters"] == {"type": "object"}
        assert instructions is None

    async def test_discovery_multiple_tools(self) -> None:
        tools_data = [
            _make_mock_tool("tool_a", "Tool A", {"type": "object"}),
            _make_mock_tool("tool_b", "Tool B", {"type": "object"}),
        ]
        mock_result = MagicMock()
        mock_result.tools = tools_data

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=_mock_init_result())
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        with (
            patch("aios.mcp.client.streamable_http_client") as mock_transport,
            patch("aios.mcp.client.ClientSession") as mock_session_cls,
        ):
            mock_transport.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tools, _instructions = await discover_mcp_tools(
                "https://mcp.example.com/", "myserver", {}
            )

        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "mcp__myserver__tool_a"
        assert tools[1]["function"]["name"] == "mcp__myserver__tool_b"

    async def test_discovery_failure_returns_empty(self) -> None:
        with patch("aios.mcp.client.streamable_http_client") as mock_transport:
            mock_transport.return_value.__aenter__ = AsyncMock(side_effect=ConnectionError("down"))
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await discover_mcp_tools("https://bad.example.com/", "bad", {})

        assert result == ([], None)

    async def test_discovery_propagates_server_instructions(self) -> None:
        """``InitializeResult.instructions`` is the standard MCP transport
        for per-server prompt affordance prose.  The harness reads it from
        ``discover_mcp_tools``'s second return slot to compose the
        per-connector system-prompt block.
        """
        mock_tool = _make_mock_tool("signal_send", "Send a Signal message", {"type": "object"})
        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(
            return_value=_mock_init_result("## Signal\n\nUse signal_send to reply.")
        )
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        with (
            patch("aios.mcp.client.streamable_http_client") as mock_transport,
            patch("aios.mcp.client.ClientSession") as mock_session_cls,
        ):
            mock_transport.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            tools, instructions = await discover_mcp_tools("https://mcp.signal/", "signal", {})

        assert len(tools) == 1
        assert instructions == "## Signal\n\nUse signal_send to reply."


# ── call_mcp_tool ─────────────────────────────────────────────────────────────


class TestCallMcpTool:
    async def test_successful_call(self) -> None:
        mock_content = MagicMock()
        mock_content.text = "Issue #42 created"
        mock_content.type = "text"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        with (
            patch("aios.mcp.client.streamable_http_client") as mock_transport,
            patch("aios.mcp.client.ClientSession") as mock_session_cls,
        ):
            mock_transport.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await call_mcp_tool(
                "https://mcp.github.com/",
                {"Authorization": "Bearer token"},
                "create_issue",
                {"title": "Test"},
            )

        assert result == {"content": "Issue #42 created"}

    async def test_error_result(self) -> None:
        mock_content = MagicMock()
        mock_content.text = "Permission denied"
        mock_content.type = "text"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = True

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        with (
            patch("aios.mcp.client.streamable_http_client") as mock_transport,
            patch("aios.mcp.client.ClientSession") as mock_session_cls,
        ):
            mock_transport.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await call_mcp_tool("https://mcp.github.com/", {}, "create_issue", {})

        assert result == {"error": "Permission denied"}

    async def test_connection_failure(self) -> None:
        with patch("aios.mcp.client.streamable_http_client") as mock_transport:
            mock_transport.return_value.__aenter__ = AsyncMock(
                side_effect=ConnectionError("server unreachable")
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await call_mcp_tool("https://bad.example.com/", {}, "tool", {})

        assert "error" in result
        assert "MCP server error" in result["error"]

    async def test_non_text_content_handled(self) -> None:
        mock_img = MagicMock()
        mock_img.type = "image"
        del mock_img.text  # no text attribute
        mock_result = MagicMock()
        mock_result.content = [mock_img]
        mock_result.isError = False

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        with (
            patch("aios.mcp.client.streamable_http_client") as mock_transport,
            patch("aios.mcp.client.ClientSession") as mock_session_cls,
        ):
            mock_transport.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )
            mock_transport.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await call_mcp_tool("https://example.com/", {}, "tool", {})

        assert result == {"content": "[image content]"}
