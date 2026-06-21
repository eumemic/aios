"""E2E: a scheduled-task (trigger) fire's sandbox carries the real secret on
egress to an allowlisted host (#884).

The aios analog of Anthropic's scheduled-deployment fires. The substrate this
gates is the landed triggers slice (#819 slice 2): a ``sandbox_command`` action
runs bash in the OWNING session's already-provisioned sandbox, and a
``workflow`` action launches a run — both inheriting the env-var credential
wiring (#877 session path / #882 run path) for free. This module proves that
inheritance end to end at the FIRE seam: drive the real trigger dispatch path
(``run_trigger_step`` → ``_run_sandbox_command`` → ``get_or_provision`` →
``build_spec_from_session`` → secret-egress proxy) and observe the
placeholder→real-secret swap actually FIRE at a recording upstream — not merely
assert the DNAT rule is present, and not merely assert the placeholder reaches
the container env.

Mirrors the run-side swap-firing legs in ``test_run_env_var_placeholder.py``
(#1160) but for the scheduled-fire origin, which has no separate credential
seam: the firing session's sandbox is ordinary session provisioning, so the
acceptance is that the SAME swap fires when the bash is reached via a trigger
fire rather than a model step.

Acceptance (issue #884): "A scheduled fire's sandbox egress to an allowlisted
host carries the real secret." The real secret must never enter the session
event surface or the trigger audit trail.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import httpx
import pytest
from pydantic import SecretStr

from aios.db import queries
from aios.harness import runtime
from aios.harness.trigger_runner import run_trigger_step
from aios.models.environments import (
    EnvironmentConfig,
    LimitedNetworking,
    NetworkingConfig,
    UnrestrictedNetworking,
)
from aios.models.triggers import (
    OneShotSource,
    SandboxCommandAction,
    TriggerCreate,
)
from aios.models.vaults import VaultCredentialCreate
from aios.sandbox import secret_egress_proxy as sep
from aios.sandbox.egress_ca import get_egress_ca
from aios.services import triggers as trig_service
from aios.services import vaults as vaults_service
from aios.services.vaults import mint_secret_placeholder
from tests.conftest import needs_docker
from tests.e2e.harness import Harness
from tests.helpers.recorder_upstream import RecorderUpstream

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"
_SECRET_NAME = "GITHUB_TOKEN"
# DNS-resolvable inside the sandbox so the DNAT sidecar can pin a ``-d <ip>``
# rule on it (the chokepoint that routes the request to the proxy); the proxy's
# *upstream* hop is then redirected to the in-process recorder, so no traffic
# ever actually reaches the real host.
_SWAP_HOST = "api.github.com"
_SWAP_SECRET = "ghp_TRIGGER_SWAP_FIRED_REAL_SECRET_DO_NOT_LEAK"

# A bash command (the sandbox_command action's payload) that drives a real
# outbound HTTPS request carrying the placeholder in an Authorization header.
# ``--resolve`` is deliberately NOT used: curl resolves the host itself, hits
# the nat-OUTPUT DNAT, and is redirected to the proxy — exercising the real
# chokepoint. The body is written to /tmp (never stdout) so the recorder's
# echo, not the swapped header, is all that could surface in the audit trail.
_SWAP_COMMAND = (
    f"curl -sS --max-time 25 -o /tmp/body "
    f'-H "Authorization: Bearer ${_SECRET_NAME}" '
    f"https://{_SWAP_HOST}/trigger-swap-probe"
)


@pytest.fixture
async def recorder_upstream() -> AsyncIterator[RecorderUpstream]:
    """A TLS recorder standing in for the credential host's real upstream.

    Presents a leaf for ``_SWAP_HOST`` minted from the worker's egress CA, so
    the proxy's upstream TLS verification (SNI + cert pinned to the real
    hostname) passes exactly as against the genuine host.
    """
    recorder = RecorderUpstream(hostname=_SWAP_HOST)
    await recorder.start()
    try:
        yield recorder
    finally:
        await recorder.stop()


def redirect_secret_egress_upstream(
    monkeypatch: pytest.MonkeyPatch, recorder: RecorderUpstream
) -> None:
    """Point the secret-egress proxy's upstream hop at ``recorder``.

    Identical to the run-side leg's redirect (``test_run_env_var_placeholder``):
    the proxy resolves the SNI host to a pinned IP and dials ``https://<ip>``
    over a freshly-verified TLS connection; we override that single hop to land
    at the recorder, leaving the placeholder→secret swap and the whole
    in-sandbox/DNAT chain untouched. Only where the bytes finally land is
    redirected, so the swap is observed against an upstream the test controls.
    """

    async def _pinned(host: str, port: int) -> str | None:
        if host == _SWAP_HOST:
            return recorder.ip
        return None

    monkeypatch.setattr(sep, "_resolve_pinned_ip", _pinned)
    monkeypatch.setattr(sep, "_UPSTREAM_PORT", recorder.port)

    orig_init = sep.SecretEgressProxy.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        verify_ctx = ssl.create_default_context()
        verify_ctx.load_verify_locations(cadata=get_egress_ca().cert_pem)
        old_client = self._client
        self._client = httpx.AsyncClient(
            timeout=old_client.timeout,
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=0),
            verify=verify_ctx,
        )
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(old_client.aclose())

    monkeypatch.setattr(sep.SecretEgressProxy, "__init__", _patched_init)


async def _provision_swap_session_and_trigger(
    pool: asyncpg.Pool[Any],
    crypto_box: Any,
    docker_harness: Harness,
    *,
    networking: NetworkingConfig,
    name: str,
) -> tuple[str, str, str]:
    """Create a credentialed session + a one-shot ``sandbox_command`` trigger.

    Returns ``(trigger_id, owner_session_id, placeholder)``. The credential's
    ``secret_value`` is the sentinel that must surface at the recorder iff the
    swap fires.
    """
    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name=name, metadata={}
    )
    cred = await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=_ACCOUNT_ID,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name=_SECRET_NAME,
            secret_value=SecretStr(_SWAP_SECRET),
            allowed_hosts=[_SWAP_HOST],
        ),
    )
    # The firing session owns the sandbox the trigger runs in — its egress is
    # ordinary session provisioning (#877). A bash tool is not required: the
    # trigger executes the command directly via the sandbox registry.
    session = await docker_harness.start(
        name,
        tools=["bash"],
        environment_config=EnvironmentConfig(networking=networking),
    )
    async with pool.acquire() as conn:
        await queries.set_session_vaults(conn, session.id, [vault.id], account_id=_ACCOUNT_ID)
        salt = await queries.get_or_create_account_placeholder_salt(conn, crypto_box, _ACCOUNT_ID)
    placeholder = mint_secret_placeholder(salt, session.id, cred.id)

    # A one-shot whose fire_at is already in the past: ``run_trigger_step``
    # treats it as a due tick fire (the substrate already enforces the at-most-
    # once delete-before-action ordering). A one-shot, not cron, so the row
    # self-deletes — keeping the test's audit assertions to a single fire.
    created = await trig_service.add_trigger(
        pool,
        session.id,
        TriggerCreate(
            name=f"swap-{name}"[:64].replace("-", "_"),
            source=OneShotSource(kind="one_shot", fire_at=datetime.now(UTC) - timedelta(seconds=1)),
            action=SandboxCommandAction(kind="sandbox_command", command=_SWAP_COMMAND),
        ),
        account_id=_ACCOUNT_ID,
    )
    return created.id, session.id, placeholder


async def _assert_trigger_swap_fired(
    docker_harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    recorder: RecorderUpstream,
    *,
    networking: NetworkingConfig,
    name: str,
) -> None:
    """Drive one scheduled fire and assert the secret-egress swap fired (#884).

    The shared body for both networking modes: provision a credentialed
    session + a due one-shot ``sandbox_command`` trigger under ``networking``,
    redirect the proxy's upstream hop to ``recorder``, fire the trigger via the
    real ``run_trigger_step`` dispatch path, and prove — at the recorder, not
    by grepping for a DNAT rule — that the placeholder→real-secret swap
    actually fired and the request completed end to end. The real secret must
    never enter the session event surface or the trigger audit trail.
    """
    pool = docker_harness._pool
    crypto_box = runtime.require_crypto_box()

    redirect_secret_egress_upstream(monkeypatch, recorder)

    trigger_id, session_id, placeholder = await _provision_swap_session_and_trigger(
        pool, crypto_box, docker_harness, networking=networking, name=name
    )

    # The one-shot failure path surfaces a synthetic wake on the owning session
    # via the stimulate spine; patch its defer out (no worker is running) so the
    # fire's own dispatch is what's under test, not the wake plumbing.
    with mock.patch("aios.harness.trigger_runner.sessions_service.stimulate", new=AsyncMock()):
        # tick fire: no trigger_run_id (one-shot due tick, not an event carrier).
        await run_trigger_step(trigger_id)

    # THE observation: the swap fired. Exactly one request landed at the
    # recorder, carrying the REAL secret in Authorization — never the
    # placeholder (which is what would arrive if the swap had silently not
    # fired). This is the runtime interaction a presence-grep of the DNAT rule
    # cannot observe.
    assert len(recorder.requests) == 1, [r.target for r in recorder.requests]
    observed = recorder.requests[0]
    assert observed.target == "/trigger-swap-probe"
    auth = observed.header("authorization")
    assert auth == f"Bearer {_SWAP_SECRET}", auth
    assert placeholder not in (auth or "")

    # The fire's audit trail recorded an ``ok`` outcome (the command exited 0:
    # curl reached the recorder and got a 200 back through the no-DROP
    # forwarding). The one-shot row self-deleted, so the terminal audit row is
    # the only persistent record.
    async with pool.acquire() as conn:
        audit_rows = await conn.fetch(
            "SELECT row_to_json(t)::text AS raw, status FROM trigger_runs t "
            "WHERE trigger_id = $1 ORDER BY started_at",
            trigger_id,
        )
        session_event_rows = await conn.fetch(
            "SELECT row_to_json(e)::text AS raw FROM events e WHERE session_id = $1 ORDER BY seq",
            session_id,
        )
    assert len(audit_rows) == 1, audit_rows
    assert audit_rows[0]["status"] == "ok", audit_rows[0]["raw"]

    # The real secret stays worker-side: it must not surface in the trigger
    # audit trail or the owning session's event surface. Only the placeholder is
    # ever container-visible.
    raw_audit = "\n".join(row["raw"] for row in audit_rows)
    raw_session_events = "\n".join(row["raw"] for row in session_event_rows)
    assert _SWAP_SECRET not in raw_audit
    assert _SWAP_SECRET not in raw_session_events


@needs_docker
async def test_trigger_swap_fires_under_limited(
    docker_harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    recorder_upstream: RecorderUpstream,
) -> None:
    """#884 acceptance — scheduled fire under the #877/#878 Limited path.

    A due one-shot ``sandbox_command`` trigger, owned by a session with an
    ``environment_variable`` credential bound under Limited networking
    (lockdown DROP + credential-host DNAT), fires the placeholder→real-secret
    swap at the recorder. This is the canonical scheduled-deployment-parity
    leg: a fired session is just a session, so the landed #877 wiring carries
    the secret to the allowlisted host at the fire seam.
    """
    await _assert_trigger_swap_fired(
        docker_harness,
        monkeypatch,
        recorder_upstream,
        networking=LimitedNetworking(type="limited", allowed_hosts=[_SWAP_HOST]),
        name="trig-swap-limited-e2e",
    )


@needs_docker
async def test_trigger_swap_fires_under_unrestricted_dnat_only(
    docker_harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    recorder_upstream: RecorderUpstream,
) -> None:
    """#884 acceptance — scheduled fire under the Unrestricted DNAT-only path.

    The Unrestricted-networking leg: the credential host's nat-OUTPUT DNAT is
    installed with the filter table at default-ACCEPT, and a scheduled fire's
    real outbound HTTPS request still fires the swap at the recorder. Pairs
    with the Limited leg so both networking modes are covered for the fire
    origin, exactly as the run origin is.
    """
    await _assert_trigger_swap_fired(
        docker_harness,
        monkeypatch,
        recorder_upstream,
        networking=UnrestrictedNetworking(type="unrestricted"),
        name="trig-swap-unrestricted-e2e",
    )
