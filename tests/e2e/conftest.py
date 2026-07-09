"""E2E test fixtures.

The ``harness`` fixture provides a :class:`~tests.e2e.harness.Harness`
instance backed by a real testcontainer Postgres with migrations applied.
Two mocks are installed for the fixture's lifetime:

1. ``litellm.acompletion`` → pops scripted responses from the harness
2. ``defer_wake`` → no-op (tests drive steps manually)

Everything else is real: the step function, async tool dispatch, event
log, context builder, and (in the ``docker_harness`` variant) real
Docker containers.

**Crash-recovery test architecture (issue #1757).** aios does not run a real
``kill -9`` worker process in CI — an audit found it would buy near-zero
marginal DB-visible coverage (Postgres commits ``append_event`` atomically
under a per-session row lock, so a real kill's ONLY unique content vs. a
faithful simulator is timing and Postgres teardown semantics, neither of
which is aios code) while requiring a permanent out-of-process harness
(subprocess worker, cross-process fake model server, dedicated DB) that has
no home (no nightly CI lane exists; the ``slow`` marker's "run nightly" is
aspirational — nothing runs ``-m slow``). Instead, crash recovery is tested at
three levels, each covering a distinct gap:

1. **Composition** — ``tests/integration/test_worker_startup_recovery.py``
   seeds the full post-kill residue (a ghost tool call, a wedged
   procrastinate ``doing`` row, a stranded model-dispatch park) simultaneously
   against a real Postgres and asserts one
   :func:`aios.harness.worker.run_startup_recovery` pass converges all of it
   in the pinned order (reap → reset in-flight harvests → wake sessions →
   wake runs). This is the boot-time SEQUENCING gap a real-kill e2e would
   have transitively covered.
2. **Running-corpse salvage** — ``tests/e2e/test_sandbox_salvage.py``'s
   ``test_running_corpse_is_salvaged_then_resumed`` is the one state ONLY a
   real SIGKILL produces that no other test covers: graceful shutdown STOPs
   every managed container before the process exits, so a crash corpse is
   normally exercised stopped; a real kill leaves it RUNNING (nothing tells
   the Docker daemon, a separate process, to stop it). Driven against the
   real daemon.
3. **The crash-state contract** — :meth:`~tests.e2e.harness.Harness.simulate_sigkill`'s
   docstring states the invariant every crash-recovery test in this suite
   relies on (a committed event prefix; procrastinate rows as-committed;
   RUNNING sandbox containers; empty process-local state) plus the checklist
   that defends against simulator drift: any NEW process-local worker state
   must ship a boot-time reset (like ``reset_inflight_harvests``, #1635) and
   a seeded crash-recovery test. Read it before adding a new in-memory
   registry to the harness.

Accepted residual risk (named, not hidden): no automated anchor exercises a
genuinely-killed process end-to-end. Mitigation is the #147 manual repro drill
(``docs/ops/``, run before promoting recovery-path changes, never promoted
into the PR-gating docker shard) plus production telemetry
(``reap_stalled_jobs`` logs a loud non-zero count on every boot after a real
death).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import socket
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
import uvicorn

from tests.e2e.harness import Harness
from tests.helpers.connections import authed_client, wait_for_health, wired_app

if TYPE_CHECKING:
    from aios.sandbox.backends.docker import DockerBackend


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    """A function-scoped asyncpg pool against the testcontainer DB.

    Promoted here from ~20 byte-identical per-file copies. Tests that
    need a pre-bootstrap (no seeded root account) DB define their own
    ``pool`` over ``aios_env_minimal`` — that local fixture overrides
    this one via standard pytest resolution (see
    ``test_accounts_bootstrap.py``).
    """
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def daemon(tmp_path: Path) -> AsyncIterator[tuple[DockerBackend, str, str, Path]]:
    """A real :class:`DockerBackend` + a unique (instance, session, workspace)
    plus container/snapshot-image cleanup.

    Shared by the real-daemon e2e sandbox tests (persistence, provision-path).
    The salvage suite defines its own differently-shaped ``daemon`` fixture,
    which overrides this one for that module (standard pytest resolution).
    """
    from aios.sandbox.backends.docker import DockerBackend
    from aios.sandbox.network import ensure_sandbox_network
    from aios.sandbox.spec import snapshot_tag

    await ensure_sandbox_network()
    backend = DockerBackend()
    instance_id = f"test_{uuid.uuid4().hex[:8]}"
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    try:
        yield backend, instance_id, session_id, workspace
    finally:
        # Remove any containers + the snapshot image this test produced.
        for ref in await backend.list_managed(instance_id=instance_id):
            await backend.force_remove(ref.sandbox_id)
        await backend.remove_image(snapshot_tag(instance_id, session_id))


_DEFAULT_DEFER_WAKE_PATCHES: tuple[str, ...] = (
    "aios.api.routers.sessions.defer_wake",
    "aios.api.routers.connectors.defer_wake",
    "aios.services.inbound.defer_wake",
)


@contextlib.asynccontextmanager
async def _live_aios_server_impl(
    *,
    defer_wake_patches: tuple[str, ...],
    pool_max_size: int,
    readiness: Callable[[str, uvicorn.Server], Awaitable[None]],
) -> AsyncIterator[str]:
    """Shared body for :func:`live_aios_server` and :func:`live_aios_server_cold`.

    The two variants differ only in their readiness probe — health-GET
    vs. polling ``server.started`` without sending any HTTP request.
    Everything else (pool, app state, uvicorn config, defer_wake mocks,
    graceful shutdown) is identical, so factoring it out keeps the two
    public entry points to a thin wrapper each.
    """
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=pool_max_size)
    app = wired_app(pool)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server.config.load()
    server.lifespan = server.config.lifespan_class(server.config)

    async def _serve() -> None:
        sock.setblocking(False)
        await server.serve(sockets=[sock])

    with contextlib.ExitStack() as patches:
        for target in defer_wake_patches:
            patches.enter_context(mock.patch(target, new_callable=mock.AsyncMock))
        serve_task = asyncio.create_task(_serve())
        try:
            url = f"http://127.0.0.1:{port}"
            await readiness(url, server)
            yield url
        finally:
            await server.shutdown()
            serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serve_task
            await pool.close()


async def _readiness_via_health(url: str, _server: uvicorn.Server) -> None:
    await wait_for_health(url)


async def _readiness_via_started_flag(
    _url: str, server: uvicorn.Server, *, deadline_s: float = 5.0
) -> None:
    """Wait until uvicorn finishes ``startup()`` without issuing any HTTP request.

    Polling ``server.started`` — set to ``True`` at the end of
    :meth:`uvicorn.Server.startup` immediately after ``create_server()``
    returns — confirms the listening socket is bound and the asyncio
    server has been created, all without sending a packet to it. The
    SSE-first-open regression (#377) reproduced exclusively when the
    first request to land on uvicorn was the SSE GET itself; any
    prior request (including a /v1/health probe) warmed the pool /
    connector init enough to mask the bug. This probe deliberately
    avoids opening any TCP connection so the test's first
    ``client.stream(...)`` is genuinely uvicorn's first request.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while not server.started:
        if loop.time() >= deadline:
            raise TimeoutError(f"uvicorn did not finish startup within {deadline_s}s")
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def live_aios_server(
    *,
    defer_wake_patches: tuple[str, ...] = _DEFAULT_DEFER_WAKE_PATCHES,
    pool_max_size: int = 4,
) -> AsyncIterator[str]:
    """Run uvicorn on a free port serving the aios app; yield base URL.

    Used by e2e tests that need a real HTTP socket (SSE streaming,
    chunked transfer, real-uvicorn behaviour) rather than the
    ``ASGITransport`` shortcut. ``defer_wake_patches`` is mocked for
    the context's lifetime — the tests drive session advancement
    directly via the harness, so the production ``defer_wake`` would
    queue jobs the test never executes. The default covers the three
    production sites; SSE-only tests that never hit the connectors
    router can pass a narrower tuple to keep the mock surface tight.

    Uvicorn's ``should_exit`` is poll-based (500 ms tick); with an
    open SSE stream the poll routinely loses the race and the test
    eats the full ``wait_for`` timeout. ``server.shutdown()`` triggers
    the graceful drain synchronously instead, then cancelling the
    serve task frees the asyncio bookkeeping.
    """
    async with _live_aios_server_impl(
        defer_wake_patches=defer_wake_patches,
        pool_max_size=pool_max_size,
        readiness=_readiness_via_health,
    ) as url:
        yield url


@contextlib.asynccontextmanager
async def live_aios_server_cold(
    *,
    defer_wake_patches: tuple[str, ...] = _DEFAULT_DEFER_WAKE_PATCHES,
    pool_max_size: int = 4,
) -> AsyncIterator[str]:
    """Like :func:`live_aios_server` but yields BEFORE any HTTP request hits uvicorn.

    The SSE first-open regression (#377) only manifests when the very
    first request landing on uvicorn is the SSE GET itself; any prior
    HTTP traffic — including a ``GET /v1/health`` readiness probe —
    triggers uvicorn's first-request initialisation and masks the bug.
    This variant waits on ``server.started`` (set at the end of
    :meth:`uvicorn.Server.startup`, after the listening socket is
    bound) instead of polling the health endpoint, so the test's first
    ``client.stream(...)`` is observably uvicorn's first request.
    """
    async with _live_aios_server_impl(
        defer_wake_patches=defer_wake_patches,
        pool_max_size=pool_max_size,
        readiness=_readiness_via_started_flag,
    ) as url:
        yield url


async def wait_for_predicate(
    predicate: Callable[[], bool | Awaitable[bool]],
    *,
    max_wait_s: float = 10.0,
    interval_s: float = 0.05,
) -> None:
    """Poll ``predicate()`` until truthy or ``max_wait_s`` elapses.
    Predicate may be sync or async."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait_s
    while loop.time() < deadline:
        result: bool | Awaitable[bool] = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        await asyncio.sleep(interval_s)
    raise AssertionError(f"predicate {predicate!r} did not become true within {max_wait_s}s")


async def _noop_defer_wake(
    pool: Any,
    session_id: str,
    *,
    account_id: str,
    cause: str = "message",
    delay_seconds: float | None = None,
) -> None:
    pass


@pytest.fixture
async def real_wake_setup(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    """Real pool against the session-migrated DB.

    Used by tests that exercise the real ``defer_wake`` and a real
    procrastinate worker — the ``harness`` fixture's ``defer_wake`` is
    a no-op and so unsuitable here. The procrastinate schema + aios
    lock-release trigger are applied once at session start by
    ``migrated_db_url``; this fixture only needs a fresh pool.
    """
    from aios.config import get_settings
    from aios.db.pool import create_pool

    pool = await create_pool(get_settings().db_url, min_size=1, max_size=8)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def open_procrastinate_app(aios_env: dict[str, str]) -> AsyncIterator[None]:
    """Open the procrastinate ``app`` against the testcontainer DB for the test.

    Provision-bearing e2e tests reach ``defer_run_wake`` (and any other
    ``app.configure_task(...).defer_async(...)`` enqueue) on the run-provision
    path. That call requires the procrastinate app to be open — in production
    the api/worker lifespan opens it (``api/app.py``, ``harness/worker.py``);
    in tests nothing does, so the enqueue raises ``procrastinate.AppNotOpen``
    at **setup**, before the test exercises the behavior under test (#1148,
    #1163).

    The module-level ``app`` singleton's connector was built at import time
    from possibly-stale settings, so — mirroring ``test_wake_lock_release_latency``
    and ``test_worker_concurrency`` — this fixture builds a fresh
    ``PsycopgConnector`` pointed at this test instance's DB, opens it, and
    swaps it in via ``replace_connector`` for the fixture's lifetime. The
    swapped connector is closed on teardown.

    Shared so every provision-bearing e2e inherits an open app
    (``docker_harness`` depends on it) rather than each test opening one
    itself.
    """
    from procrastinate import PsycopgConnector

    from aios.jobs.app import _sync_dsn
    from aios.jobs.app import app as procrastinate_app

    connector = PsycopgConnector(conninfo=_sync_dsn(aios_env["AIOS_DB_URL"]))
    await connector.open_async()
    try:
        with procrastinate_app.replace_connector(connector):
            yield
    finally:
        await connector.close_async()


@pytest.fixture
def crypto_box(aios_env: dict[str, str]) -> Any:
    """The :class:`CryptoBox` keyed by the test instance's ``AIOS_VAULT_KEY``.

    Tests that call into ``aios.services.connections.create_connection`` /
    ``set_connection_secrets`` need this — those services accept a
    ``CryptoBox`` so the in-process encryption is testable without a
    running api process.
    """
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    return CryptoBox.from_base64(get_settings().vault_key.get_secret_value())


@pytest.fixture
async def harness(aios_env: dict[str, str]) -> AsyncIterator[Harness]:
    """Function-scoped harness: real Postgres, scripted model, no Docker."""
    import aios.tools  # noqa: F401  — register built-in tools
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.harness.inflight_tool_registry import InflightToolRegistry
    from aios.tools.registry import registry
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    inflight_reg = InflightToolRegistry()

    # Save runtime globals
    prev = (
        runtime.pool,
        runtime.crypto_box,
        runtime.inflight_tool_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
        runtime.tool_provider,
    )

    # Install
    runtime.pool = pool
    runtime.crypto_box = crypto_box
    runtime.inflight_tool_registry = inflight_reg
    runtime.sandbox_registry = None  # no Docker in fast tier
    runtime.worker_id = "worker_test"
    runtime.tool_provider = SubsystemToolProvider()

    # Snapshot tool registry
    tool_snapshot = dict(registry._tools)

    h = Harness(pool, inflight_reg)

    # Install mocks at fixture scope so they cover fire-and-forget tool tasks
    async def _fake_acompletion(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return h._pop_streaming_response(**kwargs)
        return h._pop_response(**kwargs)

    def _fake_chunk_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return h._build_chunk_response(*args, **kwargs)

    with (
        mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
        mock.patch("aios.harness.completion.litellm.stream_chunk_builder", _fake_chunk_builder),
        # Since #1476 the deferral primitives live in ``aios.jobs.app`` and are
        # bound at module load into each caller's namespace, so a mock must be
        # installed where the name is *looked up*, not at the (now-defunct)
        # ``aios.services.wake`` re-export. The harness drives steps directly, so
        # the wakes this must neutralise originate in ``aios.harness.loop``
        # (step nudge / retry / auto-error wakes) and ``aios.services.sessions``
        # (invoke / message wakes); patch them where they are looked up.
        mock.patch("aios.harness.loop.defer_wake", _noop_defer_wake),
        mock.patch("aios.services.sessions.defer_wake", _noop_defer_wake),
    ):
        yield h

    # Restore
    registry._tools = tool_snapshot
    (
        runtime.pool,
        runtime.crypto_box,
        runtime.inflight_tool_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
        runtime.tool_provider,
    ) = prev
    await inflight_reg.shutdown()
    await pool.close()


@pytest.fixture
async def docker_harness(
    aios_env: dict[str, str], open_procrastinate_app: None
) -> AsyncIterator[Harness]:
    """Like ``harness`` but with a real SandboxRegistry for Docker tests.

    Depends on ``open_procrastinate_app`` so the procrastinate app is open
    for the harness's lifetime: provision-bearing run tests reach
    ``defer_run_wake`` on the run-provision path, which enqueues a
    procrastinate task and otherwise crashes at setup with
    ``procrastinate.AppNotOpen`` (#1148, #1163). Sharing it here means every
    provision-bearing e2e inherits the open app for free.

    pytest-xdist note: ``SANDBOX_NETWORK_NAME`` is a hardcoded host-global
    name (``aios-sandbox``).  Under ``-n 2``, both xdist workers' sandbox
    containers join the same bridge.  This is safe for the current test
    set — assertions are per-container (iptables rules, broker
    reachability) and don't depend on bridge composition or
    inter-container reachability.  If you add a test that asserts "this
    container is the only/first/last thing on aios-sandbox" or relies on
    cross-container DNS resolution, you'll need to either run it
    serially (``--dist=no``) or thread a per-worker network name through
    ``aios.sandbox.network``.
    """
    import aios.tools  # noqa: F401
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.harness.inflight_tool_registry import InflightToolRegistry
    from aios.sandbox.backends.docker import DockerBackend
    from aios.sandbox.network import ensure_sandbox_network
    from aios.sandbox.registry import SandboxRegistry
    from aios.sandbox.tool_broker import ToolBroker
    from aios.tools.registry import registry
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    inflight_reg = InflightToolRegistry()
    sandbox_reg = SandboxRegistry(backend=DockerBackend())
    # Mirror worker startup: create+attach the sandbox network before
    # bringing the broker up. Otherwise the very first sandbox a test
    # provisions hits ``docker run --network aios-sandbox`` against a
    # network that doesn't exist on this host yet.
    await ensure_sandbox_network()
    tool_broker = ToolBroker()
    await tool_broker.start()

    prev = (
        runtime.pool,
        runtime.crypto_box,
        runtime.inflight_tool_registry,
        runtime.sandbox_registry,
        runtime.tool_broker,
        runtime.worker_id,
        runtime.tool_provider,
    )
    runtime.pool = pool
    runtime.crypto_box = crypto_box
    runtime.inflight_tool_registry = inflight_reg
    runtime.sandbox_registry = sandbox_reg
    runtime.tool_broker = tool_broker
    runtime.worker_id = "worker_test"
    runtime.tool_provider = SubsystemToolProvider()

    tool_snapshot = dict(registry._tools)
    h = Harness(pool, inflight_reg)

    async def _fake_acompletion(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return h._pop_streaming_response(**kwargs)
        return h._pop_response(**kwargs)

    def _fake_chunk_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return h._build_chunk_response(*args, **kwargs)

    with (
        mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
        mock.patch("aios.harness.completion.litellm.stream_chunk_builder", _fake_chunk_builder),
        # See the ``harness`` fixture: since #1476 ``defer_wake`` is bound into
        # each caller's namespace, so patch the harness step call sites, not the
        # ``aios.services.wake`` re-export.
        mock.patch("aios.harness.loop.defer_wake", _noop_defer_wake),
        mock.patch("aios.services.sessions.defer_wake", _noop_defer_wake),
    ):
        yield h

    registry._tools = tool_snapshot
    (
        runtime.pool,
        runtime.crypto_box,
        runtime.inflight_tool_registry,
        runtime.sandbox_registry,
        runtime.tool_broker,
        runtime.worker_id,
        runtime.tool_provider,
    ) = prev
    await inflight_reg.shutdown()
    from tests.helpers.sandbox import purge_all_sandboxes

    await purge_all_sandboxes(sandbox_reg)
    await tool_broker.stop()
    await pool.close()


@pytest.fixture
async def http_client(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    """In-process API client wired against the testcontainer DB.

    Wakes are mocked out (no worker) so endpoints that ``defer_wake``
    (POST /messages, POST /tool-results, POST /connectors/inbound) don't
    trip on the absent procrastinate connector.
    """
    import httpx

    from aios.config import get_settings
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    app = wired_app(pool)
    # The fixture skips the API lifespan (httpx ASGITransport doesn't
    # fire startup/shutdown), so mirror the lifespan's runtime
    # registrations explicitly — the ``/context`` endpoint reaches
    # ``runtime.require_tool_provider()`` via ``compute_step_prelude``.
    prev_tool_provider = runtime.tool_provider
    runtime.tool_provider = SubsystemToolProvider()
    transport = httpx.ASGITransport(app=app)
    # Mock at every call site that imports ``defer_wake`` directly —
    # patching the source (``aios.services.wake``) is too late since the
    # importing modules already captured the unmocked reference.
    try:
        with (
            mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock),
            mock.patch("aios.services.inbound.defer_wake", new_callable=mock.AsyncMock),
        ):
            async with authed_client(
                "http://testserver",
                aios_env["AIOS_API_KEY"],
                transport=transport,
            ) as client:
                yield client
    finally:
        runtime.tool_provider = prev_tool_provider
        await pool.close()
