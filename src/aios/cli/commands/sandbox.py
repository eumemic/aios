"""Operator subcommand: ``sandbox-prewarm`` (#1348).

Bakes an operator-side *prewarm* image from the configured sandbox base
(``settings.docker_image``) with the per-deployment egress CA — and,
optionally, one environment's packages — already installed, then stamps it
with the labels the cold-start skip gate reads. Reusing such an image as
``AIOS_DOCKER_IMAGE`` removes ``install_egress_ca`` / ``install_packages``
from the cold-start hot path (a pure latency optimization; never a
correctness dependency — the skipped steps are idempotent and the gate
fails toward running them).

It is an **operator** command, NOT a client CLI command: it needs
``AIOS_EGRESS_CA_KEY`` (the CA is an HKDF subkey of it, so the bake is
structurally operator-side — public CI has no such secret) and shells out
to ``docker`` on the host. It therefore follows the operator-command
pattern (``api``/``worker``/``migrate``/``rekey`` in ``ops.py``) — no
``--url``/``--api-key``, it does not talk to the API.

What it does (synchronous, host-side):

1. ``docker run --detach`` the base — a plain run, NO managed/instance/
   session labels.
2. Build a transient :class:`SandboxHandle` and ``install_egress_ca``; if
   ``--environment`` is given, load its :class:`EnvironmentConfig` and
   ``install_packages``.
3. ``docker commit`` the container to ``--tag``, stamping BOTH
   ``BASE_IMAGE_LABEL_KEY=<base>`` AND ``PREWARM_LABEL_KEY=<base>``.
4. ``docker rm -f`` the transient container; print the tag.

It deliberately does NOT stamp ``MANAGED_LABEL_KEY`` /
``INSTANCE_LABEL_KEY`` / ``SESSION_LABEL_KEY`` — that is what keeps the
prewarm image out of the image GC's reapable set (it is treated like the
base image, an external operator-managed dependency).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from aios.models.environments import EnvironmentConfig
from aios.sandbox.backends.base import (
    BASE_IMAGE_LABEL_KEY,
    PREWARM_LABEL_KEY,
    SandboxBackend,
    SandboxBackendError,
    SandboxHandle,
)
from aios.sandbox.setup import install_egress_ca, install_packages


async def bake_prewarm_image(
    backend: SandboxBackend,
    *,
    base_image: str,
    tag: str,
    environment_id: str | None = None,
    account_id: str | None = None,
) -> str:
    """Bake a prewarm image from ``base_image`` to ``tag`` and return ``tag``.

    Pure composition of the shipped #916 snapshot/label primitive (run the
    base out-of-band, install setup, commit + stamp labels). Backend-typed so
    it is unit-testable against a fake backend with no Docker daemon.
    """
    # 1. Plain run of the base — no managed/instance/session labels. A prewarm
    #    image must NOT enter the GC's reapable set; see module docstring.
    sandbox_id = await backend.prewarm_run(base_image)
    handle = SandboxHandle(
        owner_id="prewarm",
        sandbox_id=sandbox_id,
        workspace_path=Path("/workspace"),
    )
    try:
        # 2. The idempotent setup execs being amortized into the bake.
        await install_egress_ca(backend, handle)
        if environment_id is not None:
            env_config = await _load_environment_config(environment_id, account_id)
            await install_packages(backend, handle, env_config)

        # 3. Commit + stamp BOTH labels. The container ran from ``base_image``,
        #    so that is the base ref both labels record.
        await backend.prewarm_commit(
            sandbox_id,
            tag,
            labels={
                BASE_IMAGE_LABEL_KEY: base_image,
                PREWARM_LABEL_KEY: base_image,
            },
        )
    finally:
        # 4. ``docker rm -f`` the transient container (best-effort).
        await backend.prewarm_remove(sandbox_id)
    return tag


async def _load_environment_config(
    environment_id: str, account_id: str | None
) -> EnvironmentConfig:
    """Load one environment's :class:`EnvironmentConfig` for the package bake.

    ``get_environment_config_for_id`` is account-scoped (a run can never read
    another tenant's env config), so ``--account`` is required alongside
    ``--environment``.
    """
    from aios.config import get_settings
    from aios.db import queries
    from aios.db.pool import create_pool

    if account_id is None:
        raise typer.BadParameter("--account is required when --environment is given")
    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            env_config = await queries.get_environment_config_for_id(
                conn, environment_id, account_id=account_id
            )
    finally:
        await pool.close()
    if env_config is None:
        raise typer.BadParameter(
            f"environment {environment_id!r} not found for account {account_id!r}"
        )
    return env_config


async def _run_prewarm_async(tag: str, environment_id: str | None, account_id: str | None) -> int:
    from aios.config import get_settings
    from aios.sandbox.backends.docker import DockerBackend

    base_image = get_settings().docker_image
    backend = DockerBackend()
    baked = await bake_prewarm_image(
        backend,
        base_image=base_image,
        tag=tag,
        environment_id=environment_id,
        account_id=account_id,
    )
    typer.echo(baked)
    return 0


def _run_prewarm(tag: str, environment_id: str | None, account_id: str | None) -> int:
    try:
        return asyncio.run(_run_prewarm_async(tag, environment_id, account_id))
    except SandboxBackendError as err:
        typer.echo(f"sandbox-prewarm failed: {err}", err=True)
        return 1


def register(app: typer.Typer) -> None:
    """Attach the ``sandbox-prewarm`` operator command to the root app."""

    @app.command(
        "sandbox-prewarm",
        help=(
            "Bake an operator prewarm image from the configured sandbox base "
            "(AIOS_DOCKER_IMAGE) with the egress CA (and optionally one "
            "environment's packages) pre-installed, stamped with the prewarm "
            "labels the cold-start skip gate reads. Point AIOS_DOCKER_IMAGE at "
            "the resulting tag to amortize cold-start setup."
        ),
    )
    def sandbox_prewarm(
        tag: str = typer.Option(..., "--tag", help="Image ref to commit the prewarm bake to."),
        environment: str | None = typer.Option(
            None,
            "--environment",
            help="Environment id whose packages to bake in (requires --account).",
        ),
        account: str | None = typer.Option(
            None,
            "--account",
            help="Account id that owns --environment (env config is account-scoped).",
        ),
    ) -> None:
        raise typer.Exit(_run_prewarm(tag, environment, account))
