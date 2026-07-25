"""Default-policy E2E coverage for self-URL / loopback targets (#861, PR #1931).

The #861 policy has two enforcement points, and this module proves BOTH of them
are actually armed in the E2E process under the DEFAULT configuration — i.e.
with no operator allowlist in effect:

1. **Persistence rejection** — a loopback / self-URL target cannot be *stored*.
   ``validate_outbound_target_url`` runs as a field validator on the ingress
   bodies that persist an outbound target (``mcp_servers[].url`` on agents,
   ``VaultCredentialCreate.target_url``), so the write fails loudly instead of
   latently at execution.
2. **Connection rejection** — a loopback / self-URL target cannot be *reached*.
   ``PinnedTransport`` re-resolves and validates every request's host, so even a
   target that somehow got persisted (legacy row, DNS rebind) is refused at the
   transport with no packet sent to the private address.

Why this file exists: ``tests/e2e/__init__.py`` used to set
``AIOS_OAUTH_ALLOW_INSECURE_HOSTS=127.0.0.1,localhost`` process-globally, which
silently disarmed both points for the entire E2E suite — a green run proved
nothing about the loopback path. That relaxation is gone (see
``tests/e2e/local_targets.py`` for the narrowly-scoped opt-in that replaced it);
these tests are the standing guard that it does not come back. They assert on
the DEFAULT policy explicitly and never request the opt-in fixture.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from pydantic import ValidationError as PydanticValidationError

from aios.errors import OAuthFlowError
from aios.models.agents import McpServerSpec
from aios.models.vaults import OAuthStartRequest, VaultCredentialCreate

ACCOUNT_ID = "acc_test_stub"
REDIRECT_URI = "https://console.example.com/api/auth/mcp-oauth/callback"

# Loopback / self-URL spellings that must be refused under the default policy.
# Includes the numeric encodings the 2026-07-15 adversarial review confirmed as
# bypasses (decimal / hex / short-form IPv4) plus IPv6 loopback and the
# link-local cloud-metadata address.
LOOPBACK_TARGETS = (
    "http://127.0.0.1:8080/mcp",
    "http://localhost:8080/mcp",
    "http://2130706433/mcp",
    "http://0x7f000001/mcp",
    "http://127.1/mcp",
    "http://[::1]:8080/mcp",
    "http://169.254.169.254/latest/meta-data/",
)


@pytest.fixture(autouse=True)
def _assert_default_policy(aios_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the DEFAULT policy for every test here: no operator allowlist.

    Deleting the variable (rather than trusting it to be unset) makes these
    tests independent of whatever the ambient environment does, so re-adding a
    process-global relaxation anywhere cannot make them vacuously pass.
    """
    from aios.config import get_settings

    monkeypatch.delenv("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", raising=False)
    get_settings.cache_clear()
    assert get_settings().oauth_allow_insecure_host_set == frozenset()


class TestPersistenceRejection:
    """A self-URL / loopback target cannot be STORED under the default policy."""

    @pytest.mark.parametrize("target", LOOPBACK_TARGETS)
    def test_agent_mcp_server_url_rejected(self, target: str) -> None:
        with pytest.raises(PydanticValidationError, match="private, internal, or runtime-local"):
            McpServerSpec(name="probe", url=target)

    @pytest.mark.parametrize("target", LOOPBACK_TARGETS)
    def test_vault_credential_target_url_rejected(self, target: str) -> None:
        with pytest.raises(PydanticValidationError, match="private, internal, or runtime-local"):
            VaultCredentialCreate(
                target_url=target,
                auth_type="bearer_header",
                token=SecretStr("t"),
            )

    async def test_credential_row_is_never_written(self, pool: Any, crypto_box: Any) -> None:
        """The rejection happens at the write boundary — nothing lands in the DB.

        Constructing the body raises, so ``create_vault_credential`` is never
        reachable with a loopback target; assert the table is genuinely empty
        afterwards rather than trusting the exception alone.
        """
        from aios.services import vaults as svc

        vault = await svc.create_vault(
            pool, display_name="selfurl-persist", metadata={}, account_id=ACCOUNT_ID
        )
        for target in LOOPBACK_TARGETS:
            with pytest.raises(PydanticValidationError):
                VaultCredentialCreate(
                    target_url=target, auth_type="bearer_header", token=SecretStr("t")
                )
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM vault_credentials WHERE vault_id = $1", vault.id
            )
        assert count == 0

    async def test_runtime_self_url_origin_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The runtime's OWN public API origin is a self-URL, path notwithstanding.

        This is the #861 threat in its purest form: pointing an MCP/vault target
        at the runtime's own API so it gets contacted with admin credentials.
        """
        monkeypatch.setenv("AIOS_URL", "https://runtime.example/v1")
        with pytest.raises(PydanticValidationError, match="private, internal, or runtime-local"):
            McpServerSpec(name="self", url="https://runtime.example/mcp")
        with pytest.raises(PydanticValidationError, match="private, internal, or runtime-local"):
            VaultCredentialCreate(
                target_url="https://runtime.example/anything",
                auth_type="bearer_header",
                token=SecretStr("t"),
            )


class TestConnectionRejection:
    """A self-URL / loopback target cannot be REACHED under the default policy."""

    @pytest.mark.parametrize("target", LOOPBACK_TARGETS)
    async def test_pinned_transport_refuses_to_dial(self, target: str) -> None:
        """``PinnedTransport`` fails closed before the inner transport is used.

        The inner transport is a sentinel that records any request reaching it;
        a blocked target must raise ``httpx.ConnectError`` with the sentinel
        never called — i.e. no packet is sent toward the private address.
        """
        from aios.pinned_transport import PinnedTransport

        reached: list[httpx.Request] = []

        class _Sentinel(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                reached.append(request)
                return httpx.Response(200)

        transport = PinnedTransport(inner=_Sentinel())
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.ConnectError, match="private/internal address"):
                await client.get(target)
        assert reached == []

    async def test_live_local_server_is_not_reachable_as_a_target(self) -> None:
        """A REAL, listening local server still cannot be dialed as a target.

        The strongest form of the connection assertion: the address is genuinely
        up (so a failure cannot be mistaken for "nothing was listening"), yet
        ``PinnedTransport`` still refuses it, and a direct client without the
        transport reaches the very same URL. Only the policy is in the way.
        """
        from aios.pinned_transport import PinnedTransport
        from tests.e2e.conftest import live_aios_server

        async with live_aios_server() as url:
            # Sanity: the server really is up and answering on that loopback
            # URL. Any non-5xx proves a live listener answered (the fixture
            # skips the API lifespan, so /v1/health itself may 404 — what
            # matters is that the connection succeeded).
            async with httpx.AsyncClient() as direct:
                probe = await direct.get(f"{url}/v1/health")
                assert probe.status_code < 500

            async with httpx.AsyncClient(transport=PinnedTransport()) as guarded:
                with pytest.raises(httpx.ConnectError, match="private/internal address"):
                    await guarded.get(f"{url}/v1/health")

    async def test_oauth_start_refuses_loopback_target(self, pool: Any, crypto_box: Any) -> None:
        """The OAuth connect flow's guard fails closed on loopback targets.

        Same assertion ``test_vault_oauth.py::test_rejects_insecure_target``
        makes; duplicated here against an explicitly-default policy so the
        protection is proven by a test that cannot be disarmed by an ambient
        allowlist.
        """
        from aios.services import vault_oauth as svc
        from aios.services import vaults as vaults_svc

        vault = await vaults_svc.create_vault(
            pool, display_name="selfurl-oauth", metadata={}, account_id=ACCOUNT_ID
        )
        for target in ("https://localhost/mcp", "https://127.0.0.1/mcp"):
            with pytest.raises(OAuthFlowError):
                await svc.start_oauth_flow(
                    pool,
                    crypto_box,
                    account_id=ACCOUNT_ID,
                    vault_id=str(vault.id),
                    body=OAuthStartRequest(target_url=target, redirect_uri=REDIRECT_URI),
                )


class TestOptInIsScoped:
    """The replacement opt-in relaxes policy only inside its own window."""

    async def test_allow_local_targets_window_opens_and_closes(self) -> None:
        from tests.e2e.local_targets import allow_local_targets

        # Armed by default.
        with pytest.raises(PydanticValidationError):
            McpServerSpec(name="probe", url="http://127.0.0.1:8080/mcp")

        # Relaxed inside the window, for the named host only.
        with allow_local_targets("127.0.0.1:8080"):
            assert McpServerSpec(name="probe", url="http://127.0.0.1:8080/mcp").url
            with pytest.raises(PydanticValidationError):
                McpServerSpec(name="other", url="http://10.0.0.5/mcp")

        # Re-armed on exit — the relaxation does not leak to later tests.
        with pytest.raises(PydanticValidationError):
            McpServerSpec(name="probe", url="http://127.0.0.1:8080/mcp")
