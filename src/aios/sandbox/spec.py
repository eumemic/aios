"""Build a :class:`SandboxSpec` (and friends) for a session.

The orchestrator side of provisioning. Reads the session's environment
config, materializes attached memory stores and github repositories
(starting a per-session :class:`GitProxy` if any github repos are
attached), and assembles a :class:`SandboxSpec` plus the side artifacts
the registry needs to track separately:

- The materialized echo lists (used to compute the mount snapshot the
  registry stores on the handle for drift detection).
- The :class:`EnvironmentConfig` (consumed by
  :mod:`aios.sandbox.setup` for package installation and network
  lockdown).
- The :class:`GitProxy` (if any), whose lifecycle the registry owns —
  started here so the spec builder can rewrite working-tree origins
  through the proxy URL, stopped by the registry on session release.

Pure orchestration: no Docker calls, no subprocess invocations beyond
the cache-clone helpers in :mod:`aios.sandbox.github_clone`. Whether the
returned spec is realized by a Docker backend, a host-subprocess
backend, or anything else is decided by the worker's selected backend.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from aios.config import get_settings
from aios.db import queries
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.environments import (
    RESERVED_SANDBOX_IMAGE_PREFIX,
    EnvironmentConfig,
    LimitedNetworking,
)
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.models.memory_stores import MemoryStoreResourceEcho
from aios.sandbox.backends.base import (
    BASE_IMAGE_LABEL_KEY,
    ENV_KEYS_LABEL_KEY,
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    Limited,
    Mount,
    NetworkPolicy,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.egress_ca import TRUST_STORE_ENV
from aios.sandbox.git_proxy import GitProxy
from aios.sandbox.github_clone import (
    GithubCloneError,
    ensure_cache_clone,
    ensure_session_working_tree,
)
from aios.sandbox.network import WORKER_NETWORK_ALIAS, is_running_in_container
from aios.sandbox.setup import WORKSPACE_RUNTIME_ENV
from aios.services import sessions as sessions_service
from aios.services.vaults import ResolvedEnvVarCredential, resolve_session_env_var_credentials

if TYPE_CHECKING:
    # Imported for typing only at module level: a runtime top-level import
    # would pull in ``aios.tools`` (via ``secret_egress_proxy`` →
    # ``url_safety``) whose package ``__init__`` imports ``bash``, which
    # imports back from this module — a cycle. The concrete class is bound
    # at module bottom (after every def runs) so it's still a patchable
    # ``aios.sandbox.spec.SecretEgressProxy`` attribute.
    from aios.sandbox.secret_egress_proxy import SecretEgressProxy

log = get_logger("aios.sandbox.spec")

# Path inside the sandbox container where the worker's tool-broker UDS
# is bind-mounted when ``settings.tool_broker_socket_path`` is set. The
# in-sandbox ``bin/tool`` CLI reads ``TOOL_BROKER_URL`` which the worker
# sets to ``unix://`` + this path. The directory matches the well-known
# location for ephemeral host runtime sockets so it doesn't collide
# with any sandbox workspace path.
TOOL_BROKER_SOCKET_SANDBOX_PATH = "/var/run/aios/tool-broker.sock"

# Path inside the sandbox container where the per-session
# TOOL_BROKER_SECRET is written and bind-mounted (read-only) when UDS
# transport is in use. The in-sandbox ``bin/tool`` CLI reads this file
# as a fallback when ``TOOL_BROKER_SECRET`` isn't in the environment.
TOOL_BROKER_SECRET_SANDBOX_PATH = "/var/run/aios/tool-broker-secret"


def snapshot_tag(deployment: str, session_id: str) -> str:
    """The deterministic snapshot ref for a session (durable session sandboxes).

    ``aios-sbx-{deployment}-{session_id.lower()}:latest`` — a single path
    component, so ``_is_registry_image`` returns False and the ``--pull
    always`` logic never reaches the registry for a snapshot. ULID lowercasing
    is bijective, so the mapping is collision-free.

    ``deployment`` is the **deployment namespace** (the ``instance_id`` in v1),
    kept conceptually distinct from the *host id* (``snapshot_host``): the ref
    must be a pure function of (deployment, session), NEVER of which worker
    committed it. If a future multi-host deployment minted refs from a
    per-worker id, every cross-host handoff would change a session's ref and
    reopen the first-commit crash window (§5.11 invariant 7). The reserved
    prefix this produces (``aios-sbx-``) is gated against tenant ``image``
    overrides at the API boundary and the spec-build choke point.
    """
    return f"{RESERVED_SANDBOX_IMAGE_PREFIX}{deployment}-{session_id.lower()}:latest"


def cleanup_session_secret_file(session_id: str, tool_socket_host_path: Path | None) -> None:
    """Best-effort removal of the per-session secret file written by
    ``_assemble_plan`` when UDS transport is enabled. Idempotent — safe
    to call when no file exists or when UDS was never configured. Logs
    but does not raise on permission / IO errors so a single failing
    session can't abort cleanup for the rest (e.g. inside
    ``SandboxRegistry.release_all``)."""
    if tool_socket_host_path is None:
        return
    secret_path = tool_socket_host_path.parent / f"{session_id}.secret"
    try:
        secret_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning(
            "sandbox.tool_broker_secret_cleanup_failed",
            session_id=session_id,
            path=str(secret_path),
            error=str(exc),
        )


@dataclass(frozen=True, slots=True)
class ProvisioningPlan:
    """Everything :class:`SandboxRegistry` needs to provision a session.

    The registry consumes this in three steps: (1) call
    ``backend.create(plan.spec)`` to bring the sandbox up, (2) run the
    setup steps from :mod:`aios.sandbox.setup` against the resulting
    handle, (3) record ``plan.git_proxy`` (if any) for cleanup at
    release time. The mount snapshot used by the drift detector lives
    on ``plan.spec`` and is stamped onto the handle by the backend.
    """

    spec: SandboxSpec
    env_config: EnvironmentConfig | None
    memory_echoes: list[MemoryStoreResourceEcho]
    github_echoes: list[GithubRepositoryResourceEcho]
    git_proxy: GitProxy | None
    # Decrypted env-var credentials + minted placeholders (#873). Alive
    # only for the provision call — the plan is dropped after
    # ``_provision`` consumes it. Materialization (placeholder env
    # injection) and the request-time placeholder→secret map are the
    # follow-up slices (#874, #876).
    env_var_credentials: tuple[ResolvedEnvVarCredential, ...]
    # Per-session :class:`SecretEgressProxy` (#877), started here when the
    # session has env-var credentials. Like ``git_proxy``, the registry owns
    # its lifecycle: it records the proxy at provision time and stops it on
    # release / recycle. ``None`` when the session has no env-var creds.
    # INERT until #878 wires nat DNAT egress routing into it.
    secret_proxy: SecretEgressProxy | None = None


class _EnvVarCredentialLike(Protocol):
    """The structural shape the env-var drift fold needs from a credential.

    Provision passes :class:`ResolvedEnvVarCredential`; the per-step probe
    passes :class:`aios.db.queries.EnvVarCredentialEcho`. Both satisfy this,
    so :func:`mount_snapshot_from_echoes` folds them identically (constraint A).
    """

    @property
    def credential_id(self) -> str: ...

    @property
    def updated_at(self) -> datetime: ...


def mount_snapshot_from_echoes(
    memory_echoes: list[MemoryStoreResourceEcho] | tuple[MemoryStoreResourceEcho, ...],
    github_echoes: list[GithubRepositoryResourceEcho] | tuple[GithubRepositoryResourceEcho, ...],
    env_var_credentials: Sequence[_EnvVarCredentialLike] = (),
) -> frozenset[tuple[str, ...]]:
    """The set of inputs that determines the spec's mount/credential surface.

    Order-independent so rank reorders don't trigger spurious recycles.
    Each tuple is type-prefixed so memory, github, and vault-credential
    namespaces can't collide. Both the github tuple and the env-var tuple
    include ``updated_at`` so a token/secret rotation (which bumps
    ``updated_at``) propagates to a sandbox recycle (#877).

    ``env_var_credentials`` is structural — provision passes
    :class:`ResolvedEnvVarCredential`, the per-step drift probe passes
    :class:`aios.db.queries.EnvVarCredentialEcho`. Both fold to the SAME
    ``(VAULT_CREDENTIAL, credential_id, updated_at)`` element, so the
    provision-time snapshot stamped on the handle matches the step-time
    echo set (no spurious first-step recycle).
    """
    from aios.ids import GITHUB_REPOSITORY, MEMORY_STORE, VAULT_CREDENTIAL

    items: set[tuple[str, ...]] = set()
    for m in memory_echoes:
        items.add((MEMORY_STORE, m.memory_store_id, m.name, m.access))
    for g in github_echoes:
        items.add((GITHUB_REPOSITORY, g.id, g.mount_path, g.updated_at.isoformat()))
    for c in env_var_credentials:
        items.add((VAULT_CREDENTIAL, c.credential_id, c.updated_at.isoformat()))
    return frozenset(items)


async def _load_environment_config(session_id: str, *, account_id: str) -> EnvironmentConfig | None:
    """Load the environment config for a session from the DB.

    Returns ``None`` if the session or environment doesn't exist (shouldn't
    happen in normal flow, but callers handle it gracefully).
    """

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_environment_config_for_session(
            conn, session_id, account_id=account_id
        )


async def resolve_bash_timeout_ceiling(session_id: str) -> int:
    """Return the maximum allowed bash-tool timeout (seconds) for a session.

    A session bound to an environment whose
    :attr:`EnvironmentConfig.bash_timeout_seconds` is set uses that
    environment's ceiling; otherwise the worker-global
    ``settings.bash_default_timeout_seconds`` (issue #725). The agent may
    still request a shorter per-call timeout — this is the cap it is
    clamped to, never a floor.

    Total by design: any failure to load the environment config (no
    environment attached, session/env not found, transient DB hiccup)
    falls back to the global default rather than raising, so a bash call
    never fails to *run* merely because the per-env override couldn't be
    resolved. The bound it falls back to is the conservative (smaller-or-
    equal) global one, so the fallback can't widen the ceiling beyond
    what the operator configured globally.
    """
    settings = get_settings()
    default = settings.bash_default_timeout_seconds
    try:
        account_id = await sessions_service.load_session_account_id(
            runtime.require_pool(), session_id
        )
        env_config = await _load_environment_config(session_id, account_id=account_id)
    except Exception as exc:  # total by contract (see docstring)
        log.warning(
            "sandbox.bash_timeout_resolve_failed",
            session_id=session_id,
            error=str(exc),
        )
        return default
    if env_config is not None and env_config.bash_timeout_seconds is not None:
        return env_config.bash_timeout_seconds
    return default


async def _load_session_provisioning(
    session_id: str, *, account_id: str
) -> tuple[str, dict[str, str], int, str | None]:
    """Load workspace path, env, ``spec_version``, and the snapshot pointer
    (``snapshot_ref``) from the session row in one query."""

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_session_provisioning(conn, session_id, account_id=account_id)


async def _materialize_memory_mounts(
    session_id: str,
    *,
    account_id: str,
) -> list[MemoryStoreResourceEcho]:
    """Materialize each attached memory store and return the echo list.

    Returns the per-store metadata used to assemble mounts after this
    call returns. A separate helper (vs inline in ``build_spec``) keeps
    the unit-test mocking surface tight — tests mock this single
    function and don't need a live pool.
    """
    from aios.sandbox.memory_mounts import materialize_store_to_host

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        echoes = await queries.list_session_memory_store_echoes(
            conn, session_id, account_id=account_id
        )
        for echo in echoes:
            await materialize_store_to_host(
                conn, store_id=echo.memory_store_id, account_id=account_id
            )
    return list(echoes)


async def _materialize_env_var_credentials(
    session_id: str,
    *,
    account_id: str,
) -> tuple[ResolvedEnvVarCredential, ...]:
    """Resolve + decrypt the session's env-var credentials and mint their
    placeholders (#873).

    Runs BEFORE :func:`_materialize_github_clones` in
    :func:`build_spec_from_session` by design: this step holds no
    host-side resources, so its deliberate fail-hard on a corrupt blob
    (one bad credential aborts the provision) raises while there is
    nothing to clean up — after the clones step a raise would have to
    unwind a running GitProxy.
    """
    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()
    async with pool.acquire() as conn:
        resolved = await resolve_session_env_var_credentials(
            conn, crypto_box, session_id, account_id=account_id
        )
    return tuple(resolved)


async def _materialize_github_clones(
    session_id: str,
    *,
    account_id: str,
) -> tuple[list[GithubRepositoryResourceEcho], GitProxy | None]:
    """Decrypt tokens, start a per-session :class:`GitProxy`, then run
    the host-side clone for each attached github_repository — rewriting
    each working tree's ``origin`` to the proxy URL so the bind-mounted
    ``.git/config`` inside the sandbox carries no credential.

    Returns the materialized echo list and the proxy (or ``None`` when
    the session has no github_repository attachments).

    Per-repo clone failures are non-fatal — they log + append a
    lifecycle event and drop the repo from the materialized list (its
    working tree won't be bind-mounted). The proxy stays alive for
    sibling repos. Per-clone wall-clock is bounded by
    ``Settings.github_clone_session_timeout_seconds`` (issue #697).
    """
    from aios.sandbox.git_proxy import repo_key
    from aios.services import github_repositories as github_repo_service

    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()

    # Load echoes and decrypt all tokens up front so the proxy can be
    # initialized with the full token map before any clone runs.
    async with pool.acquire() as conn:
        echoes = await queries.list_session_github_repo_echoes(
            conn, session_id, account_id=account_id
        )
        if not echoes:
            return [], None
        echo_tokens: list[tuple[GithubRepositoryResourceEcho, str]] = []
        repos_map: dict[str, str] = {}
        for echo in echoes:
            token = await github_repo_service.get_session_token(
                conn, crypto_box, session_id, echo.id, account_id=account_id
            )
            echo_tokens.append((echo, token))
            repos_map[repo_key(echo.url)] = token

    proxy = GitProxy(repos_map)
    await proxy.start()

    # The per-repo except below catches only GithubCloneError. Anything
    # else (OSError from a full disk during mkdir/rmtree in
    # ensure_session_working_tree, CancelledError on step interrupt,
    # asyncpg failure during the failure-event append) escapes — and
    # the proxy is already running. Without this outer guard the
    # uvicorn server, httpx client, bound TCP port, and in-memory
    # token map leak for the worker's lifetime. Mirrors the cleanup
    # pattern at build_spec_from_session below.
    try:
        materialized: list[GithubRepositoryResourceEcho] = []
        failures: list[tuple[GithubRepositoryResourceEcho, GithubCloneError]] = []
        for echo, token in echo_tokens:
            try:
                cache_dir = await ensure_cache_clone(echo.url, token)
                await ensure_session_working_tree(
                    session_id=session_id,
                    resource_id=echo.id,
                    repo_url=echo.url,
                    token=token,
                    cache_dir=cache_dir,
                    proxy_url=proxy.proxy_url(echo.url, host=WORKER_NETWORK_ALIAS),
                    git_user_name=echo.git_user_name,
                    git_user_email=echo.git_user_email,
                )
            except GithubCloneError as err:
                failures.append((echo, err))
                continue
            materialized.append(echo)

        # Log + append failure events outside the conn block so we don't
        # hold one connection while acquiring a second from the same pool.
        if failures:
            for failed_echo, failure in failures:
                log.warning(
                    "sandbox.github_clone_failed",
                    session_id=session_id,
                    resource_id=failed_echo.id,
                    repo_url=failed_echo.url,
                    error=str(failure),
                )
                await sessions_service.append_event(
                    pool,
                    session_id,
                    "lifecycle",
                    {
                        "event": "github_clone_failed",
                        "resource_id": failed_echo.id,
                        "repo_url": failed_echo.url,
                        "message": str(failure),
                    },
                    account_id=account_id,
                )
        return materialized, proxy
    except BaseException:
        try:
            await proxy.stop()
        except Exception as cleanup_err:
            log.warning(
                "sandbox.git_proxy_cleanup_after_materialize_failure",
                session_id=session_id,
                error=str(cleanup_err),
            )
        raise


def _network_policy_from_config(env_config: EnvironmentConfig | None) -> NetworkPolicy:
    """Translate the environment config's networking field to a
    backend-agnostic :class:`NetworkPolicy`."""
    networking = env_config.networking if env_config else None
    if isinstance(networking, LimitedNetworking):
        return Limited(allowed_hosts=frozenset(networking.allowed_hosts))
    return Unrestricted()


async def build_spec_from_session(session_id: str) -> ProvisioningPlan:
    """Assemble a :class:`ProvisioningPlan` for ``session_id``.

    Reads the session row, loads the environment config, materializes
    memory stores and github clones (starting a GitProxy if any github
    repos are attached), mints a per-session MCP-broker secret, then
    assembles the :class:`SandboxSpec` and related artifacts.

    On any failure after the first host resource is allocated — the
    GitProxy started by the clones step, the secret-egress proxy, or the
    broker secret — every allocation made so far is cleaned up before the
    exception propagates so the ports, token map, and secret map don't
    leak for the worker's lifetime. A single ``try`` envelope spans from
    the clones step through plan assembly so no allocation sits in an
    unguarded window (the reserved-image gate and broker registration
    both raise inside it).
    """
    from aios.harness import runtime
    from aios.sandbox.volumes import ensure_workspace_path, validate_workspace_path

    settings = get_settings()
    account_id = await sessions_service.load_session_account_id(runtime.require_pool(), session_id)
    raw_path, session_env, spec_version, snapshot_ref = await _load_session_provisioning(
        session_id, account_id=account_id
    )
    # Re-validate at the bind-mount boundary: ``validate_workspace_path``
    # also runs at session-create time, but a symlink swap on any
    # directory component of ``raw_path`` between then and now would
    # let ``ensure_workspace_path``'s ``Path.resolve()`` dereference
    # to an out-of-jail target.  Re-running the check immediately
    # before the bind-mount closes that TOCTOU window.
    #
    # Passing ``session_id`` enables the legacy carve-out for pre-#409
    # sessions whose row still holds ``<workspace_root>/<session_id>``
    # (no per-tenant subdir).  Without it those sessions cannot
    # cold-start a sandbox post-#590 (issue #626).
    validate_workspace_path(raw_path, account_id, session_id=session_id)
    workspace_path = ensure_workspace_path(raw_path)
    env_config = await _load_environment_config(session_id, account_id=account_id)
    memory_echoes = await _materialize_memory_mounts(session_id, account_id=account_id)
    # Resolve + decrypt env-var credentials BEFORE any host resource is
    # allocated (#873): a corrupt blob fails hard here with nothing to
    # clean up, so this step stays OUTSIDE the cleanup envelope below.
    env_var_credentials = await _materialize_env_var_credentials(session_id, account_id=account_id)

    tool_broker = runtime.require_tool_broker()
    tool_socket_host_path = settings.tool_broker_socket_path
    # Initialized to None before the cleanup envelope so the ``except``
    # can reference them whether or not they were allocated when a raise
    # fired. ``git_proxy`` is assigned by the clones step (first host
    # allocation), ``secret_proxy`` shortly after.
    git_proxy: GitProxy | None = None
    secret_proxy: SecretEgressProxy | None = None
    broker_registered = False
    try:
        github_echoes, git_proxy = await _materialize_github_clones(
            session_id, account_id=account_id
        )
        # Start the per-session secret-egress proxy when the session has
        # env-var credentials (#877). Inert until #878 routes egress through
        # it; the registry owns its lifecycle from the returned plan,
        # mirroring git_proxy.
        if env_var_credentials:
            secret_proxy = SecretEgressProxy(env_var_credentials)
            await secret_proxy.start()

        tool_broker_secret = secrets.token_urlsafe(32)
        tool_broker.register_session(session_id, tool_broker_secret)
        broker_registered = True
        # Prefer the UDS transport when the operator configured one: it
        # works in deployments where the worker's TCP port isn't reachable
        # from the sandbox network (e.g. Coolify production). Otherwise
        # fall back to the ephemeral TCP URL.
        if tool_socket_host_path is not None:
            tool_broker_url = f"unix://{TOOL_BROKER_SOCKET_SANDBOX_PATH}"
        else:
            tool_broker_url = f"http://{WORKER_NETWORK_ALIAS}:{tool_broker.port}"

        # Per-environment image override (#724): a session bound to an
        # environment with ``image`` set provisions from that image; unset
        # falls back to the worker-global ``settings.docker_image``. This is
        # the only resolution point — everything downstream (backend create,
        # the --pull decision) consumes ``spec.image`` blindly.
        image = env_config.image if (env_config and env_config.image) else settings.docker_image
        # Cross-tenant snapshot gate (durable session sandboxes, §5.8): this is
        # the spec-build choke point ALL writers traverse, including direct SQL —
        # the boundary layer of the two-layer gate (the other being the pydantic
        # ``image`` validator). Reject any image in the reserved snapshot
        # namespace so a tenant can't point ``env_config.image`` at another
        # session's snapshot tag (``aios-sbx-<victim>``) and mount its root FS +
        # baked secrets. Case-insensitive (Docker lowercases refs).
        if image.lower().startswith(RESERVED_SANDBOX_IMAGE_PREFIX):
            raise ValueError(
                f"environment image {image!r} uses the reserved "
                f"{RESERVED_SANDBOX_IMAGE_PREFIX!r} prefix (reserved for per-session "
                "snapshot images); refusing to provision"
            )
        # Per-session snapshot budget (durable session sandboxes, §5.7):
        # environment override wins; otherwise the worker-global default
        # (4 GiB unless the operator set AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES).
        snapshot_budget_bytes = (
            env_config.snapshot_budget_bytes
            if (env_config and env_config.snapshot_budget_bytes is not None)
            else settings.sandbox_snapshot_budget_bytes
        )

        return _assemble_plan(
            session_id=session_id,
            instance_id=settings.instance_id,
            image=image,
            workspace_path=workspace_path,
            snapshot_ref=snapshot_ref,
            snapshot_budget_bytes=snapshot_budget_bytes,
            env_config=env_config,
            session_env=session_env,
            memory_echoes=memory_echoes,
            github_echoes=github_echoes,
            git_proxy=git_proxy,
            tool_broker_url=tool_broker_url,
            tool_broker_secret=tool_broker_secret,
            tool_socket_host_path=tool_socket_host_path,
            spec_version=spec_version,
            env_var_credentials=env_var_credentials,
            secret_proxy=secret_proxy,
        )
    except BaseException:
        # Every host resource allocated above holds worker-process state
        # (ports, token maps, secret-to-session bindings). Anything that
        # raises before we hand back a plan must clean up everything
        # allocated so far on the way out, otherwise state leaks for the
        # worker's lifetime.
        if broker_registered:
            tool_broker.unregister_session(session_id)
            cleanup_session_secret_file(session_id, tool_socket_host_path)
        if git_proxy is not None:
            try:
                await git_proxy.stop()
            except Exception as cleanup_err:
                log.warning(
                    "sandbox.git_proxy_cleanup_after_spec_failure",
                    session_id=session_id,
                    error=str(cleanup_err),
                )
        if secret_proxy is not None:
            try:
                await secret_proxy.stop()
            except Exception as cleanup_err:
                log.warning(
                    "sandbox.secret_proxy_cleanup_after_spec_failure",
                    session_id=session_id,
                    error=str(cleanup_err),
                )
        raise


def _assemble_plan(
    *,
    session_id: str,
    instance_id: str,
    image: str,
    workspace_path: Path,
    env_config: EnvironmentConfig | None,
    session_env: dict[str, str],
    memory_echoes: list[MemoryStoreResourceEcho],
    github_echoes: list[GithubRepositoryResourceEcho],
    git_proxy: GitProxy | None,
    tool_broker_url: str,
    tool_broker_secret: str,
    tool_socket_host_path: Path | None = None,
    snapshot_ref: str | None = None,
    snapshot_budget_bytes: int | None = None,
    spec_version: int = 0,
    env_var_credentials: tuple[ResolvedEnvVarCredential, ...] = (),
    secret_proxy: SecretEgressProxy | None = None,
) -> ProvisioningPlan:
    """Pure assembly of the plan from already-materialized inputs.

    When ``tool_socket_host_path`` is set, a read-write bind mount of
    the broker socket file is appended at
    ``TOOL_BROKER_SOCKET_SANDBOX_PATH``; the caller is expected to have
    set ``tool_broker_url`` to the matching ``unix://`` URL.
    """
    from aios.sandbox.volumes import (
        ensure_session_attachments_dir,
        ensure_session_uploads_dir,
        memory_store_host_dir,
        session_repo_working_tree_dir,
    )

    # Operator-supplied env as one layer (environment config + per-session
    # overrides, session winning) so the vault placeholders below can be
    # checked against it for shadowing.
    operator_env = {
        **(env_config.env if env_config and env_config.env else {}),
        **session_env,
    }
    # Vault env-var credential placeholders (#874): the session's
    # ``environment_variable`` secrets surface in the container as
    # ``secret_name=<opaque placeholder>``. ONLY the placeholder enters
    # the container — the decrypted secret stays on
    # ``plan.env_var_credentials`` in worker memory for the egress swap
    # (#876) and never touches ``merged_env``, the spec, or any log.
    # A vaulted secret outranks an operator/session var of the same name:
    # a deliberate credential beats a loose env var, and it keeps a
    # plaintext value an operator mistakenly set under that name out of
    # the container. The clash is logged so the override is never silent.
    placeholder_env = {cred.secret_name: cred.placeholder for cred in env_var_credentials}
    shadowed = sorted(operator_env.keys() & placeholder_env.keys())
    if shadowed:
        log.warning(
            "sandbox.vault_placeholder_shadows_env",
            session_id=session_id,
            secret_names=shadowed,
        )

    merged_env: dict[str, str] = {
        **WORKSPACE_RUNTIME_ENV,
        # Trust-store defaults precede operator env so a custom image
        # (#724) with a non-Debian CA layout can override them.
        **TRUST_STORE_ENV,
        **operator_env,
        **placeholder_env,
        # Trailers last so nothing shadows them — a placeholder can't reach
        # them anyway (secret_name can't be a reserved key, models/vaults.py).
        "TOOL_BROKER_URL": tool_broker_url,
        "TOOL_BROKER_SECRET": tool_broker_secret,
        # Scheduled-tasks escalation (#636): bash inside a cron sandbox
        # POSTs to ``$TOOL_BROKER_URL/v1/$TOOL_BROKER_SECRET/sessions/messages``
        # to wake the session. AIOS_SESSION_ID is also exposed so the
        # bash can name its own session in logs / messages.
        "AIOS_SESSION_ID": session_id,
    }

    # Bind the per-session attachments and uploads dirs at every provision
    # so an inbound landing — or a ``POST /v1/sessions/<id>/files`` arriving
    # — during a running step is visible to the live sandbox.  Docker doesn't
    # update binds in place; this works because the bind exposes the
    # *directory* (not a file snapshot), and any new file staged into it
    # appears through the existing kernel namespace bind without re-mounting.
    attachments_path = ensure_session_attachments_dir(session_id)
    uploads_path = ensure_session_uploads_dir(session_id)

    extra_mounts: list[Mount] = [
        Mount(host_path=attachments_path, sandbox_path="/mnt/attachments", read_only=True),
        Mount(host_path=uploads_path, sandbox_path="/mnt/uploads", read_only=True),
    ]

    # Bind-mount each attached memory store. Read-only attaches make the
    # kernel reject writes (bash + tools alike); read-write is the
    # default. The host source is shared across all attached sessions,
    # so a tool/API write in one session is visible to all.
    for memory_echo in memory_echoes:
        host_dir = memory_store_host_dir(memory_echo.memory_store_id)
        extra_mounts.append(
            Mount(
                host_path=host_dir,
                sandbox_path=f"/mnt/memory/{memory_echo.name}",
                read_only=memory_echo.access == "read_only",
            )
        )

    for github_echo in github_echoes:
        repo_host_dir = session_repo_working_tree_dir(session_id, github_echo.id)
        extra_mounts.append(
            Mount(
                host_path=repo_host_dir,
                sandbox_path=github_echo.mount_path,
                read_only=False,
            )
        )

    if tool_socket_host_path is not None:
        # Read-write so the sandbox can write request bytes through the
        # bound socket file. The kernel routes the actual I/O through
        # the AF_UNIX endpoint regardless of file mode, but Docker
        # still applies the read_only bit to the inode itself.
        extra_mounts.append(
            Mount(
                host_path=tool_socket_host_path,
                sandbox_path=TOOL_BROKER_SOCKET_SANDBOX_PATH,
                read_only=False,
            )
        )
        # Write the per-session secret to a sibling file and bind-mount it
        # read-only. The in-sandbox ``bin/tool`` CLI reads this file as a
        # fallback when ``TOOL_BROKER_SECRET`` isn't in the environment
        # (stale container reuse, bash scripts that strip env, etc. — see
        # issue #698).
        #
        # Use ``os.open`` with explicit 0o600 mode so the create + perms
        # are applied in one syscall — a separate ``write_text`` +
        # ``chmod`` would leave a brief umask-derived window where, under
        # ``umask 0``, the file would be world-writable. 0o600 is tighter
        # than the prior 0o644 since the bind-mounted file is read by
        # root inside the sandbox; no need for group/world read on host.
        secret_host_path = tool_socket_host_path.parent / f"{session_id}.secret"
        fd = os.open(secret_host_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, tool_broker_secret.encode("utf-8"))
        finally:
            os.close(fd)
        extra_mounts.append(
            Mount(
                host_path=secret_host_path,
                sandbox_path=TOOL_BROKER_SECRET_SANDBOX_PATH,
                read_only=True,
            )
        )

    labels = {
        MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
        INSTANCE_LABEL_KEY: instance_id,
        SESSION_LABEL_KEY: session_id,
        # Durable session sandboxes (§5.1), inherited into committed images.
        # ``env_keys`` lists the NAMES of every run-injected env var so the
        # commit-time scrub empties exactly that set (never the base image's
        # PATH/HOME — those aren't here). Every name listed is re-injected via
        # ``docker run --env`` at resume, so emptying it in the image config
        # only sanitizes the artifact (no value leaks to ``docker save`` /
        # layer inspect) without affecting the resumed runtime. ``base_image``
        # records the chain root for resume base-drift detection and
        # unique-bytes accounting.
        ENV_KEYS_LABEL_KEY: ",".join(sorted(merged_env)),
        BASE_IMAGE_LABEL_KEY: image,
    }

    snapshot = mount_snapshot_from_echoes(memory_echoes, github_echoes, env_var_credentials)
    settings = get_settings()
    spec = SandboxSpec(
        session_id=session_id,
        instance_id=instance_id,
        workspace=Mount(host_path=workspace_path, sandbox_path="/workspace", read_only=False),
        extra_mounts=tuple(extra_mounts),
        environment=merged_env,
        labels=labels,
        network_policy=_network_policy_from_config(env_config),
        host_gateway_alias=None if is_running_in_container() else WORKER_NETWORK_ALIAS,
        image=image,
        # Resume source (durable session sandboxes): the raw DB snapshot
        # pointer. The registry's provision path resolves it through the store
        # (verified-negative, base-drift checked) and replaces it with the
        # locally-runnable tag — or clears it to None on a detected reset —
        # before ``backend.create``. ``None`` ⇒ cold start.
        snapshot_image=snapshot_ref,
        mount_snapshot=snapshot,
        # Provision-time ``sessions.spec_version`` snapshot (issue #713):
        # the backend copies it onto the handle so the registry's warm-hit
        # probe can detect a between-steps resource mutation and recycle.
        spec_version=spec_version,
        # Resource caps come from deployment-wide settings (multi-tenancy
        # hardening — #367 PR 9). A future revision will let an account
        # override these via the management API; the spec-layer plumbing
        # is what makes that override-point cheap to add.
        cpu_quota=settings.sandbox_cpu_quota,
        memory_bytes=settings.sandbox_memory_bytes,
        pids_limit=settings.sandbox_pids_limit,
        # Per-session snapshot budget (durable session sandboxes, §5.7):
        # already resolved by the caller (environment override → global
        # default). The backend stamps it onto the handle as
        # ``disk_limit_bytes``; the release path passes it back to
        # ``backend.snapshot`` as the flatten trigger.
        snapshot_budget_bytes=snapshot_budget_bytes,
        # Authored seccomp deny-list (#807). Always set from settings (never
        # left at the dataclass "unconfined" fallback) so the backend ships
        # the deny-list profile by default; the literal "unconfined" only
        # appears via the AIOS_SANDBOX_SECCOMP_PROFILE emergency override.
        seccomp_profile=settings.sandbox_seccomp_profile,
    )

    return ProvisioningPlan(
        spec=spec,
        env_config=env_config,
        memory_echoes=memory_echoes,
        github_echoes=github_echoes,
        git_proxy=git_proxy,
        env_var_credentials=env_var_credentials,
        secret_proxy=secret_proxy,
    )


# Runtime binding for ``SecretEgressProxy`` (see the TYPE_CHECKING note at the
# top). Imported at module bottom — after every def has run — so it resolves
# the spec↔tools cycle while still exposing a patchable
# ``aios.sandbox.spec.SecretEgressProxy`` attribute for unit tests.
from aios.sandbox.secret_egress_proxy import SecretEgressProxy  # noqa: E402
