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

from dataclasses import dataclass
from pathlib import Path

from aios.config import get_settings
from aios.db import queries
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.models.memory_stores import MemoryStoreResourceEcho
from aios.sandbox.backends.base import (
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
from aios.sandbox.git_proxy import GitProxy
from aios.sandbox.github_clone import (
    GithubCloneError,
    ensure_cache_clone,
    ensure_session_working_tree,
)
from aios.sandbox.setup import WORKSPACE_RUNTIME_ENV
from aios.services import sessions as sessions_service

log = get_logger("aios.sandbox.spec")


# Hostname the sandbox uses to reach the host-side credential proxy.
# Docker maps this to the host gateway IP via ``--add-host``; on Docker
# Desktop the name auto-resolves but the explicit alias is required on
# Linux. The registry threads the same string through the iptables script
# so limited-networking sessions can still reach the proxy.
PROXY_HOST_ALIAS = "host.docker.internal"


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


def mount_snapshot_from_echoes(
    memory_echoes: list[MemoryStoreResourceEcho] | tuple[MemoryStoreResourceEcho, ...],
    github_echoes: list[GithubRepositoryResourceEcho] | tuple[GithubRepositoryResourceEcho, ...],
) -> frozenset[tuple[str, ...]]:
    """The set of inputs that determines the spec's mount list.

    Order-independent so rank reorders don't trigger spurious recycles.
    Each tuple is type-prefixed so memory and github namespaces can't
    collide. The github tuple includes ``updated_at`` so token rotation
    (which bumps ``updated_at``) propagates to a sandbox recycle.
    """
    from aios.ids import GITHUB_REPOSITORY, MEMORY_STORE

    items: set[tuple[str, ...]] = set()
    for m in memory_echoes:
        items.add((MEMORY_STORE, m.memory_store_id, m.name, m.access))
    for g in github_echoes:
        items.add((GITHUB_REPOSITORY, g.id, g.mount_path, g.updated_at.isoformat()))
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


async def _load_session_provisioning(
    session_id: str, *, account_id: str
) -> tuple[str, dict[str, str]]:
    """Load workspace path and env from the session row in one query."""

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
            await materialize_store_to_host(conn, store_id=echo.memory_store_id)
    return list(echoes)


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
    sibling repos.
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
                proxy_url=proxy.proxy_url(echo.url, host=PROXY_HOST_ALIAS),
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


def _network_policy_from_config(env_config: EnvironmentConfig | None) -> NetworkPolicy:
    """Translate the environment config's networking field to a
    backend-agnostic :class:`NetworkPolicy`."""
    networking = env_config.networking if env_config else None
    if isinstance(networking, LimitedNetworking):
        return Limited(allowed_hosts=frozenset(networking.allowed_hosts))
    # Today the only non-limited option is unrestricted; ``Disabled`` is
    # reserved for future use (no current EnvironmentConfig produces it).
    return Unrestricted()


async def build_spec_from_session(session_id: str) -> ProvisioningPlan:
    """Assemble a :class:`ProvisioningPlan` for ``session_id``.

    Reads the session row, loads the environment config, materializes
    memory stores and github clones (starting a GitProxy if any github
    repos are attached), then assembles the :class:`SandboxSpec` and
    related artifacts.

    On any failure after the GitProxy is started, the proxy is stopped
    before the exception propagates so the port + token map don't leak
    for the worker's lifetime.
    """
    from aios.sandbox.volumes import ensure_workspace_path

    settings = get_settings()
    account_id = await sessions_service.load_session_account_id(runtime.require_pool(), session_id)
    raw_path, session_env = await _load_session_provisioning(session_id, account_id=account_id)
    workspace_path = ensure_workspace_path(raw_path)
    env_config = await _load_environment_config(session_id, account_id=account_id)
    memory_echoes = await _materialize_memory_mounts(session_id, account_id=account_id)
    github_echoes, git_proxy = await _materialize_github_clones(session_id, account_id=account_id)

    try:
        return _assemble_plan(
            session_id=session_id,
            instance_id=settings.instance_id,
            image=settings.docker_image,
            workspace_path=workspace_path,
            env_config=env_config,
            session_env=session_env,
            memory_echoes=memory_echoes,
            github_echoes=github_echoes,
            git_proxy=git_proxy,
        )
    except BaseException:
        # The proxy holds an open port + in-memory token map. Anything
        # that raises before we hand back a plan must stop the proxy on
        # the way out, otherwise the port and tokens leak for the
        # worker's lifetime.
        if git_proxy is not None:
            try:
                await git_proxy.stop()
            except Exception as cleanup_err:
                log.warning(
                    "sandbox.git_proxy_cleanup_after_spec_failure",
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
) -> ProvisioningPlan:
    """Pure assembly of the plan from already-materialized inputs."""
    from aios.sandbox.volumes import (
        ensure_session_attachments_dir,
        ensure_session_uploads_dir,
        memory_store_host_dir,
        session_repo_working_tree_dir,
    )

    merged_env: dict[str, str] = {
        **WORKSPACE_RUNTIME_ENV,
        **(env_config.env if env_config and env_config.env else {}),
        **session_env,
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

    labels = {
        MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
        INSTANCE_LABEL_KEY: instance_id,
        SESSION_LABEL_KEY: session_id,
    }

    snapshot = mount_snapshot_from_echoes(memory_echoes, github_echoes)
    settings = get_settings()
    spec = SandboxSpec(
        session_id=session_id,
        instance_id=instance_id,
        workspace=Mount(host_path=workspace_path, sandbox_path="/workspace", read_only=False),
        extra_mounts=tuple(extra_mounts),
        environment=merged_env,
        labels=labels,
        network_policy=_network_policy_from_config(env_config),
        host_gateway_alias=PROXY_HOST_ALIAS if git_proxy is not None else None,
        image=image,
        mount_snapshot=snapshot,
        # Resource caps come from deployment-wide settings (multi-tenancy
        # hardening — #367 PR 9). A future revision will let an account
        # override these via the management API; the spec-layer plumbing
        # is what makes that override-point cheap to add.
        cpu_quota=settings.sandbox_cpu_quota,
        memory_bytes=settings.sandbox_memory_bytes,
        pids_limit=settings.sandbox_pids_limit,
    )

    return ProvisioningPlan(
        spec=spec,
        env_config=env_config,
        memory_echoes=memory_echoes,
        github_echoes=github_echoes,
        git_proxy=git_proxy,
    )
