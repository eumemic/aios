"""Regression: API lifespan must clean up ALL resources on pre-yield startup failure.

If any step between pool creation and ``yield`` raises, the lifespan must:
  1. Restore runtime globals (``crypto_box``, ``tool_provider``) if changed.
  2. Close procrastinate if it was opened.
  3. Close the asyncpg pool.

These tests exercise the real lifespan by injecting failures at
``_await_retirements_admissible`` and at earlier points, then asserting
that every resource opened before the failure is cleaned up and every
runtime global is restored.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime


class _BootGateBoom(Exception):
    """Injected failure at the boot-admission gate."""


def _make_fake_pool() -> MagicMock:
    """Build a MagicMock that behaves like an asyncpg.Pool."""
    pool = MagicMock()
    pool.close = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock(fetchval=AsyncMock()))
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool


@pytest.fixture(autouse=True)
def _save_runtime_globals():
    """Snapshot and restore runtime globals around each test."""
    orig_crypto = runtime.crypto_box
    orig_tp = runtime.tool_provider
    yield
    runtime.crypto_box = orig_crypto
    runtime.tool_provider = orig_tp


def _build_app_with_patches(extra_patches: dict[str, Any] | None = None):
    """Call ``create_app()`` with everything DB/MCP-related mocked out.

    Returns ``(app, fake_pool, fake_procrastinate)`` so the caller can
    assert against the mocks.
    """
    from aios.api.app import create_app

    fake_pool = _make_fake_pool()
    fake_procrastinate = MagicMock()
    fake_procrastinate.open_async = AsyncMock()
    fake_procrastinate.close_async = AsyncMock()

    base_patches: dict[str, Any] = {
        "aios.api.app.create_pool": AsyncMock(return_value=fake_pool),
        "aios.api.app.queries.audit_credentialless_root": AsyncMock(),
        "aios.api.app.procrastinate_app": fake_procrastinate,
        # The workspace-root validator is a late import inside the lifespan;
        # patch the module it's imported from.
        "aios.sandbox.workspace_root_startup.validate_workspace_root_against_sessions": AsyncMock(),
        # MCP mount tries to iterate routes and build tool schemas — stub it.
        "aios.api.app._mount_mcp": lambda app: None,
    }
    if extra_patches:
        base_patches.update(extra_patches)

    ctxs = [patch(k, v) for k, v in base_patches.items()]
    for c in ctxs:
        c.start()
    try:
        app = create_app()
    finally:
        for c in ctxs:
            c.stop()

    return app, fake_pool, fake_procrastinate


async def test_lifespan_cleans_up_on_retirements_failure() -> None:
    """Inject failure at ``_await_retirements_admissible`` and verify full cleanup.

    After pool creation AND procrastinate open AND runtime-global mutation,
    a failure at the boot-admission gate must:
      - close procrastinate
      - close the pool
      - restore runtime.crypto_box and runtime.tool_provider
    """
    sentinel_crypto = runtime.crypto_box
    sentinel_tp = runtime.tool_provider

    app, fake_pool, fake_procrastinate = _build_app_with_patches()

    with (
        patch(
            "aios.api.app._await_retirements_admissible",
            new=AsyncMock(side_effect=_BootGateBoom("injected")),
        ),
        patch("aios.api.app.procrastinate_app", fake_procrastinate),
        patch("aios.api.app.create_pool", AsyncMock(return_value=fake_pool)),
        patch("aios.api.app.queries.audit_credentialless_root", AsyncMock()),
        patch(
            "aios.sandbox.workspace_root_startup.validate_workspace_root_against_sessions",
            AsyncMock(),
        ),
        pytest.raises(_BootGateBoom),
    ):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan should not have reached yield")

    # Pool must be closed.
    fake_pool.close.assert_awaited_once()
    # Procrastinate must be closed (it was opened before the failure point).
    fake_procrastinate.close_async.assert_awaited_once()
    # Runtime globals must be restored to their pre-lifespan values.
    assert runtime.crypto_box is sentinel_crypto, (
        "runtime.crypto_box not restored after startup failure"
    )
    assert runtime.tool_provider is sentinel_tp, (
        "runtime.tool_provider not restored after startup failure"
    )


async def test_lifespan_pool_only_on_early_failure() -> None:
    """If startup fails BEFORE procrastinate opens, only pool is closed."""
    fake_pool = _make_fake_pool()

    app, _, _ = _build_app_with_patches()

    with (
        patch("aios.api.app.create_pool", AsyncMock(return_value=fake_pool)),
        patch("aios.api.app.queries.audit_credentialless_root", AsyncMock()),
        patch(
            "aios.sandbox.workspace_root_startup.validate_workspace_root_against_sessions",
            AsyncMock(side_effect=_BootGateBoom("early")),
        ),
        pytest.raises(_BootGateBoom),
    ):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan should not have reached yield")

    fake_pool.close.assert_awaited_once()
