"""E2E: workflow-run bash sees env-var credential placeholders, not secrets.

Restores the run-side acceptance leg from #882: a workflow run with an
``environment_variable`` vault credential bound provisions a real Docker sandbox,
materializes ``SECRET_NAME=<placeholder>``, and journals only that placeholder in the
bash result. The plaintext secret must not enter the model/session event surface or
the run journal.

This module also carries the §3.6/R6 **functional swap-firing e2e** (#1160) that
gates the #1158 deploy. The placeholder-materialization legs above prove the
*placeholder* reaches the container; the swap-firing legs below drive a **real
outbound HTTPS request** through the sandbox and observe the placeholder→real-secret
swap actually fire at a recording upstream — not merely assert the DNAT rule is
present. Both the #1158 Unrestricted DNAT-only path (the first to install a
nat-OUTPUT DNAT with the filter table at default-ACCEPT) and the already-merged
#877/#878 Limited path are exercised, under prod's runtime (``runc``, the e2e
default; see ``AIOS_SANDBOX_RUNTIME``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import httpx
import pytest
from pydantic import SecretStr

from aios.db import queries
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.models.environments import (
    EnvironmentConfig,
    LimitedNetworking,
    NetworkingConfig,
    UnrestrictedNetworking,
)
from aios.models.vaults import VaultCredentialCreate
from aios.sandbox import secret_egress_proxy as sep
from aios.sandbox.egress_ca import get_egress_ca
from aios.services import environments as environments_service
from aios.services import vaults as vaults_service
from aios.services.vaults import mint_secret_placeholder, resolve_run_env_var_credentials
from aios.workflows import run_tools
from aios.workflows import service as workflows_service
from aios.workflows.step import run_workflow_step
from tests.conftest import needs_docker
from tests.e2e.harness import Harness
from tests.helpers.recorder_upstream import RecorderUpstream

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"
_SECRET_NAME = "GITHUB_TOKEN"
_SENTINEL_SECRET = "ghp_RUN_SENTINEL_PLAINTEXT_DO_NOT_LEAK"
_ALLOWED_HOST = "api.github.com"
_SCRIPT = f"""async def main(input):
    return await tool('bash', {{"command": 'printf "%s" "${_SECRET_NAME}"'}})
"""

# Swap-firing credential host. Must be DNS-resolvable inside the sandbox so the
# DNAT sidecar can pin a ``-d <ip>`` rule on it (the chokepoint that routes the
# request to the proxy); the proxy's *upstream* hop is then redirected to the
# in-process recorder (see ``redirect_secret_egress_upstream``), so no traffic
# ever actually reaches the real host.
_SWAP_HOST = "api.github.com"
_SWAP_SECRET = "ghp_SWAP_FIRED_REAL_SECRET_DO_NOT_LEAK"
# A run script that drives a real outbound HTTPS request carrying the
# placeholder in an Authorization header. ``--resolve`` is deliberately NOT
# used: curl resolves the host itself, hits the nat-OUTPUT DNAT, and is
# redirected to the proxy — exercising the real chokepoint. ``-w`` echoes the
# upstream status so the run output proves the request completed end to end.
_SWAP_SCRIPT = f"""async def main(input):
    return await tool('bash', {{"command": 'curl -sS --max-time 25 -o /tmp/body \
-w "HTTP_STATUS=%{{http_code}}" -H "Authorization: Bearer ${_SECRET_NAME}" \
https://{_SWAP_HOST}/swap-probe; echo; cat /tmp/body'}})
"""


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

    The proxy resolves the SNI host to a pinned IP and dials ``https://<ip>:443``
    over a freshly-verified TLS connection. We override that single hop:

    * ``_resolve_pinned_ip`` returns the recorder's loopback IP (the real
      resolver fail-closes on a loopback address, so we replace it wholesale)
      and ``_UPSTREAM_PORT`` is repointed at the recorder's ephemeral port;
    * the proxy's upstream httpx client is rebuilt with a trust store that
      *adds* the worker's egress CA, so verifying the recorder's CA-minted leaf
      (pinned to the real SNI host, exactly as the proxy already does) succeeds
      against our stand-in. In prod the upstream is the genuine publicly-trusted
      host; here it is the recorder presenting the same hostname under the
      egress CA — so the swap path, SNI, and cert-pinning stay real.

    The placeholder→secret swap and the whole in-sandbox/DNAT chain are
    untouched; only where the bytes finally land is redirected, so the swap is
    observed against an upstream the test controls.
    """

    async def _pinned(host: str, port: int) -> str | None:
        if host == _SWAP_HOST:
            return recorder.ip
        return None

    monkeypatch.setattr(sep, "_resolve_pinned_ip", _pinned)
    monkeypatch.setattr(sep, "_UPSTREAM_PORT", recorder.port)

    # Rebuild the proxy's upstream client to trust the egress CA on top of the
    # system store. ``ssl.create_default_context`` keeps the production verify
    # posture (hostname check + system roots); ``load_verify_locations`` folds
    # in the CA the recorder's leaf is signed by. We wrap __init__ so every
    # proxy the registry constructs during the run picks this up.
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
        # The original __init__ already opened a client; close it so it doesn't
        # leak. Schedule the close on the running loop (best-effort).
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(old_client.aclose())

    monkeypatch.setattr(sep.SecretEgressProxy, "__init__", _patched_init)


async def _provision_swap_run(
    pool: asyncpg.Pool[Any],
    crypto_box: Any,
    *,
    networking: NetworkingConfig,
    name: str,
) -> tuple[str, str]:
    """Create a vault credential + workflow + run wired for the swap probe.

    Returns ``(run_id, placeholder)``. The credential's ``secret_value`` is the
    sentinel that must surface at the recorder iff the swap fires.
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
    env = await environments_service.create_environment(
        pool,
        account_id=_ACCOUNT_ID,
        name=name,
        config=EnvironmentConfig(networking=networking),
    )
    async with pool.acquire() as conn:
        workflow = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT_ID,
            name=name,
            script=_SWAP_SCRIPT,
            tools=[ToolSpec(type="bash")],
        )

    run = await workflows_service.create_run(
        pool,
        account_id=_ACCOUNT_ID,
        workflow_id=workflow.id,
        environment_id=env.id,
        vault_ids=[vault.id],
    )
    async with pool.acquire() as conn:
        salt = await queries.get_or_create_account_placeholder_salt(conn, crypto_box, _ACCOUNT_ID)
    placeholder = mint_secret_placeholder(salt, run.id, cred.id)
    return run.id, placeholder


async def _drive_run_to_completion(pool: asyncpg.Pool[Any], run_id: str) -> None:
    for _ in range(10):
        await run_workflow_step(run_id)
        pending = [task for (rid, _), task in run_tools._INFLIGHT.items() if rid == run_id]
        if pending:
            await asyncio.gather(*pending, return_exceptions=False)
        async with pool.acquire() as conn:
            run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        if run.status == "completed":
            return
    raise AssertionError("workflow run did not complete")


@needs_docker
async def test_run_bash_env_var_placeholder_round_trip(docker_harness: Harness) -> None:
    pool = docker_harness._pool
    crypto_box = runtime.require_crypto_box()

    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="run-envvar-e2e", metadata={}
    )
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=_ACCOUNT_ID,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name=_SECRET_NAME,
            secret_value=SecretStr(_SENTINEL_SECRET),
            allowed_hosts=[_ALLOWED_HOST],
        ),
    )
    env = await environments_service.create_environment(
        pool,
        account_id=_ACCOUNT_ID,
        name="run-envvar-e2e",
        config=EnvironmentConfig(
            networking=LimitedNetworking(type="limited", allowed_hosts=[_ALLOWED_HOST]),
        ),
    )
    async with pool.acquire() as conn:
        workflow = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT_ID,
            name="run-envvar-e2e",
            script=_SCRIPT,
            tools=[ToolSpec(type="bash")],
        )

    with (
        mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.workflows.run_sandbox.defer_run_wake", new=AsyncMock()),
    ):
        run = await workflows_service.create_run(
            pool,
            account_id=_ACCOUNT_ID,
            workflow_id=workflow.id,
            environment_id=env.id,
            vault_ids=[vault.id],
        )
        async with pool.acquire() as conn:
            resolved = await resolve_run_env_var_credentials(
                conn, crypto_box, run.id, account_id=_ACCOUNT_ID
            )
        assert len(resolved) == 1
        placeholder = resolved[0].placeholder

        await _drive_run_to_completion(pool, run.id)

    async with pool.acquire() as conn:
        completed = await wf_queries.get_run_for_step(conn, run.id)
        run_event_rows = await conn.fetch(
            "SELECT row_to_json(e)::text AS raw FROM wf_run_events e WHERE run_id = $1 ORDER BY seq",
            run.id,
        )
        signal_rows = await conn.fetch(
            "SELECT row_to_json(s)::text AS raw FROM wf_run_signals s WHERE run_id = $1 ORDER BY delivered_at",
            run.id,
        )
        session_event_rows = await conn.fetch(
            """
            SELECT row_to_json(e)::text AS raw
              FROM events e
              JOIN sessions s ON s.id = e.session_id
             WHERE s.parent_run_id = $1 OR e.session_id = $1
             ORDER BY e.seq
            """,
            run.id,
        )

    assert completed is not None and completed.status == "completed"
    assert completed.output["stdout"] == placeholder
    assert completed.output["stdout"] != _SENTINEL_SECRET

    raw_session_events = "\n".join(row["raw"] for row in session_event_rows)
    raw_run_events = "\n".join(row["raw"] for row in run_event_rows)
    raw_run_signals = "\n".join(row["raw"] for row in signal_rows)
    assert _SENTINEL_SECRET not in raw_session_events
    assert _SENTINEL_SECRET not in raw_run_events
    assert _SENTINEL_SECRET not in raw_run_signals
    assert placeholder in raw_run_events


async def _assert_swap_fired_e2e(
    docker_harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    recorder: RecorderUpstream,
    *,
    networking: NetworkingConfig,
    name: str,
) -> None:
    """Drive one functional swap-firing run and assert the swap fired (#1160).

    The shared body for both the Unrestricted (#1158 DNAT-only) and Limited
    (#877/#878) legs: provision a credentialed run under ``networking``, redirect
    the proxy's upstream hop to ``recorder``, drive a real in-sandbox ``curl``
    carrying the placeholder, and prove — at the recorder, not by grepping for a
    DNAT rule — that the placeholder→real-secret swap actually fired and the
    request completed end to end (HTTP 200 back through the no-DROP forwarding).
    The real secret must never enter the run journal/event surface.
    """
    pool = docker_harness._pool
    crypto_box = runtime.require_crypto_box()

    redirect_secret_egress_upstream(monkeypatch, recorder)

    run_id, placeholder = await _provision_swap_run(
        pool, crypto_box, networking=networking, name=name
    )

    with (
        mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.workflows.run_sandbox.defer_run_wake", new=AsyncMock()),
    ):
        await _drive_run_to_completion(pool, run_id)

    async with pool.acquire() as conn:
        completed = await wf_queries.get_run_for_step(conn, run_id)
        run_event_rows = await conn.fetch(
            "SELECT row_to_json(e)::text AS raw FROM wf_run_events e WHERE run_id = $1 ORDER BY seq",
            run_id,
        )
        signal_rows = await conn.fetch(
            "SELECT row_to_json(s)::text AS raw FROM wf_run_signals s "
            "WHERE run_id = $1 ORDER BY delivered_at",
            run_id,
        )

    assert completed is not None and completed.status == "completed"
    stdout = str(completed.output.get("stdout", ""))
    # The request reached the recorder upstream and got a real HTTP response
    # back through the no-DROP forwarding path — this is the runtime interaction
    # (#1158's first nat-OUTPUT DNAT with filter at default-ACCEPT) that a
    # presence-grep of the DNAT rule cannot observe.
    assert f"HTTP_STATUS={recorder.response_status}" in stdout, stdout
    assert json.loads(recorder.response_body)["recorder"] == "ok"

    # THE observation: the swap fired. Exactly one request landed at the
    # recorder, carrying the REAL secret in Authorization — never the
    # placeholder (which is what would arrive if the swap had silently not
    # fired, e.g. a DNAT rule present against a shadow table while no rule is
    # live).
    assert len(recorder.requests) == 1, [r.target for r in recorder.requests]
    observed = recorder.requests[0]
    assert observed.target == "/swap-probe"
    auth = observed.header("authorization")
    assert auth == f"Bearer {_SWAP_SECRET}", auth
    assert placeholder not in (auth or "")

    # The real secret stays worker-side: it must not surface in the run journal,
    # the run-event surface, or the signal surface. Only the placeholder is ever
    # journaled (the curl wrote the body to /tmp, never to stdout — the recorder
    # echoes a fixed body, not the Authorization header).
    raw_run_events = "\n".join(row["raw"] for row in run_event_rows)
    raw_run_signals = "\n".join(row["raw"] for row in signal_rows)
    assert _SWAP_SECRET not in raw_run_events
    assert _SWAP_SECRET not in raw_run_signals
    assert _SWAP_SECRET not in stdout


@needs_docker
async def test_run_swap_fires_under_unrestricted_dnat_only(
    docker_harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    recorder_upstream: RecorderUpstream,
) -> None:
    """§3.6/R6 functional swap-firing e2e — the #1158 Unrestricted DNAT-only path.

    The deploy-gating leg (#1160): a credentialed run provisioned under
    **Unrestricted** networking installs the nat-OUTPUT DNAT alone with the
    filter table at **default-ACCEPT** (the first path to do so), then a real
    outbound HTTPS request fires the placeholder→real-secret swap at the
    recorder. Observing the swap at runtime — not grepping for the DNAT rule —
    is the proof the merge shipped without.
    """
    await _assert_swap_fired_e2e(
        docker_harness,
        monkeypatch,
        recorder_upstream,
        networking=UnrestrictedNetworking(type="unrestricted"),
        name="run-swap-unrestricted-e2e",
    )


@needs_docker
async def test_run_swap_fires_under_limited(
    docker_harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    recorder_upstream: RecorderUpstream,
) -> None:
    """§3.6/R6 functional swap-firing e2e — back-fill for the #877/#878 Limited path.

    The merged Limited path (lockdown DROP + credential-host DNAT) never had a
    functional swap-firing proof either; this back-fills it with the same
    recorder observation, so both networking modes are covered before deploy.
    """
    await _assert_swap_fired_e2e(
        docker_harness,
        monkeypatch,
        recorder_upstream,
        networking=LimitedNetworking(type="limited", allowed_hosts=[_SWAP_HOST]),
        name="run-swap-limited-e2e",
    )
