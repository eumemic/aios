"""Integration coverage for the workflow-binding vision-gate fix (#2358 follow-up).

The unit tests in ``test_browser_tools.py`` / ``test_read_image.py`` mock
``_resolve_capability_model`` to assert the trigger fix. These tests drive the
REAL resolver against a testcontainer Postgres: a real ``workflow:<id>`` row
with a declared ``output_model``, a real ``get_session_model`` returning the
binding string, and the real ``_resolve_capability_model`` DB lookup — so the
end-to-end resolution path is exercised, not a mocked seam.

Two shapes:

* Trigger (Section E) — invoke ``browser_screenshot_handler`` against a real
  workflow-bound session. ``get_session_model`` returns the raw
  ``workflow:<id>`` string from the session row; ``_resolve_capability_model``
  resolves it to the workflow's declared ``output_model``; the vision gate
  then keys on the INNER model. A text-only ``output_model`` degrades to a
  text marker; a vision-capable one inlines the ``image_url`` part.
* Self-heal (Section F) — manually append a persisted ``image_url`` tool-result
  event to a workflow-bound (text-only inner model) session's log, then run the
  REAL ``compose_step_context`` (the composer ``run_session_step`` calls) and
  assert the build-time ``_strip_image_parts_for_non_vision_model`` pass strips
  the part from the built messages — self-healing an already-wedged session.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from aios.config import get_settings
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime, vision
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.models.agents import ToolSpec
from aios.sandbox.browser_protocol import BrowserResponse
from aios.sandbox.volumes import ensure_browser_plane_dir
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.tools import browser as browser_mod
from aios.tools.registry import ToolResult
from tests.helpers.images import valid_png_bytes

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_wfscreenshot"
_ENV = "env_wfscreenshot"

# The workflow's inner script returns a trivial assistant turn (no tool calls);
# the screenshot tool is invoked by the OUTER handler directly in-test, so the
# inner run's output shape only needs to be a valid bound-workflow return.
_INNER_SCRIPT = (
    "async def main(input):\n"
    "    return {'content': 'inner answer', 'tool_calls': [], 'finish_reason': 'stop'}\n"
)


class _EmptyToolProvider:
    async def list_tools_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> list[dict[str, Any]]:
        return []

    async def list_capabilities_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> dict[str, Any]:
        return {}


@pytest.fixture
async def wf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool on ``runtime.pool`` + an inflight registry + a seeded root tenant.

    Mirrors ``mwf_runtime`` in ``test_model_workflow_park_sweep.py`` but does
    not patch the workflow park layer — these tests drive the screenshot
    handler / composer directly, not the full ``run_session_step`` park path.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool = runtime.pool
    prev_reg = runtime.inflight_tool_registry
    prev_tp = runtime.tool_provider
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    runtime.tool_provider = _EmptyToolProvider()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'wf-screenshot-root')",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'wf-screenshot-env', '{}'::jsonb, $2)",
                _ENV,
                _ACCOUNT,
            )
        # Register the inner models' vision capability so the gate resolves
        # without hitting litellm's catalog (the models are test fixtures).
        vision._VISION_OVERRIDES["wf/text-only"] = False
        vision._VISION_OVERRIDES["wf/vision"] = True
        yield pool
    finally:
        runtime.pool = prev_pool
        runtime.inflight_tool_registry = prev_reg
        runtime.tool_provider = prev_tp
        vision._VISION_OVERRIDES.pop("wf/text-only", None)
        vision._VISION_OVERRIDES.pop("wf/vision", None)
        await pool.close()


async def _make_bound_agent_and_session(
    pool: asyncpg.Pool[Any], *, output_model: str
) -> tuple[str, str]:
    """Create a workflow with ``output_model``, an agent bound to
    ``workflow:<id>`` (with ``browser_screenshot`` enabled), and a session.

    Returns ``(session_id, workflow_id)``.
    """
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT,
            name=f"inner-{output_model}",
            script=_INNER_SCRIPT,
            output_model=output_model,
        )
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACCOUNT,
        name=f"wf-screenshot-agent-{output_model[-1:]}",
        model=f"workflow:{wf.id}",
        system="",
        tools=[ToolSpec(type="browser_screenshot")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.get_environment(pool, _ENV, account_id=_ACCOUNT)
    session = await sessions_service.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title="wf-screenshot",
        metadata={},
        account_id=_ACCOUNT,
    )
    return session.id, wf.id


def _patch_driver_and_grant(
    tmp_path: Path, account_id: str, png: bytes, pool: asyncpg.Pool[Any]
) -> Any:
    """Patch ``driver_call`` to return a ``shot_path`` whose plane file holds
    ``png``, patch the grant re-check, and stand up ``runtime.sandbox_registry``
    (the handler's ``_invoke_driver`` calls ``require_sandbox_registry()``)."""
    settings = get_settings()
    settings.workspace_root = tmp_path
    plane = ensure_browser_plane_dir(account_id) / "shots"
    plane.mkdir(parents=True, exist_ok=True)
    (plane / "shot.png").write_bytes(png)

    # The handler resolves the session's account id via the DB (not mocked), so
    # the load_session_account_id seam runs against the real pool.
    from aios.sandbox.browser_protocol import BrowserRequest, BrowserResponse

    async def fake_driver_call(
        registry: Any, acct: str, request: BrowserRequest, *, timeout_s: float
    ) -> BrowserResponse:
        return BrowserResponse.model_validate(
            {
                "ok": True,
                "boot": "01BOOT",
                "epoch": 4,
                "url": "https://example.com/checkout",
                "title": "Checkout",
                "snapshot": "- button 'Place order' [ref=e12]",
                "duration_ms": 812,
                "shot_path": "shots/shot.png",
            }
        )

    fake_runtime = MagicMock()
    fake_runtime.require_pool.return_value = pool
    fake_runtime.require_sandbox_registry.return_value = MagicMock()
    return mock.patch.multiple(
        browser_mod,
        driver_call=fake_driver_call,
        _check_arm_granted=AsyncMock(),
        runtime=fake_runtime,
    )


# ─── Section E: trigger — real DB-backed workflow resolution ──────────────


async def test_text_only_output_model_degrades_screenshot_to_text(
    wf_runtime: asyncpg.Pool[Any], tmp_path: Path
) -> None:
    """The handler keyed on the raw ``workflow:<id>`` string pre-fix; with the
    fix it resolves the binding to the declared ``output_model`` (a real DB
    lookup) and degrades to a text marker when the inner model is text-only.

    This drives the REAL ``get_session_model`` (returns the binding string
    from the session row) and the REAL ``_resolve_capability_model`` (looks up
    the workflow's ``output_model`` from the workflow row) — no mocking of the
    resolver, only of the browser driver.
    """
    pool = wf_runtime
    session_id, _ = await _make_bound_agent_and_session(pool, output_model="wf/text-only")
    png = valid_png_bytes()

    with _patch_driver_and_grant(tmp_path, _ACCOUNT, png, pool):
        result = await browser_mod.browser_screenshot_handler(session_id, {})

    assert isinstance(result, ToolResult)
    assert isinstance(result.content, str), (
        f"text-only workflow binding must degrade to a marker; got {result.content!r}"
    )
    assert "does not support image input" in result.content
    assert "wf/text-only" in result.content  # the RESOLVED model name, not the workflow: string
    assert "workflow:" not in result.content


async def test_vision_output_model_inlines_screenshot(
    wf_runtime: asyncpg.Pool[Any], tmp_path: Path
) -> None:
    """A workflow binding whose declared ``output_model`` is vision-capable
    inlines the screenshot (no over-degradation) — driven against the real
    DB-backed resolution."""
    pool = wf_runtime
    session_id, _ = await _make_bound_agent_and_session(pool, output_model="wf/vision")
    png = valid_png_bytes()

    with _patch_driver_and_grant(tmp_path, _ACCOUNT, png, pool):
        result = await browser_mod.browser_screenshot_handler(session_id, {})

    assert isinstance(result, ToolResult)
    assert isinstance(result.content, list)
    assert result.content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode()}"},
    }


# ─── Section F: self-heal — real composer strips a persisted image_url part ──


async def test_build_time_pass_strips_persisted_image_for_text_only_workflow(
    wf_runtime: asyncpg.Pool[Any], tmp_path: Path
) -> None:
    """An already-wedged session (a persisted ``image_url`` tool result frozen in
    the event log) self-heals on the next build: the real ``compose_step_context``
    resolves the workflow binding to its text-only ``output_model`` and the
    build-time ``_strip_image_parts_for_non_vision_model`` pass downgrades the
    ``image_url`` part to a text marker in the built messages.

    Drives the REAL composer (``compose_step_context``, the function
    ``run_session_step`` calls) against a real workflow-bound session with a
    manually-appended ``image_url`` tool-result event — the exact replay shape
    a wedged session presents on every wake.
    """
    pool = wf_runtime
    session_id, _ = await _make_bound_agent_and_session(pool, output_model="wf/text-only")

    # Seed a user message so the session has a stimulus and an event log.
    await sessions_service.append_user_message(
        pool, session_id, "show me the page", account_id=_ACCOUNT
    )

    # Append the wedging tool result: an assistant turn proposing a
    # browser_screenshot tool_call, then a tool result whose content is a list
    # with an image_url data URI — the exact shape a pre-fix inline produced.
    png = valid_png_bytes()
    url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
    from aios.db import queries as db_queries

    async with pool.acquire() as conn:
        await db_queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session_id,
            kind="message",
            data={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_wedge",
                        "type": "function",
                        "function": {"name": "browser_screenshot", "arguments": "{}"},
                    }
                ],
                "finish_reason": "tool_calls",
                "model": "workflow:wf",
            },
        )
        await db_queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session_id,
            kind="message",
            data={
                "role": "tool",
                "tool_call_id": "call_wedge",
                "name": "browser_screenshot",
                "content": [
                    {"type": "text", "text": "Screenshot: shot.png (image/png, 1.0KB)"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        )

    # Run the REAL composer the step loop runs. The loop resolves the workflow
    # binding to its declared ``output_model`` via ``_resolve_capability_model``
    # (loop.py:947) and passes it as ``capability_model``; mirror that here so
    # the build-time pass keys on the RESOLVED inner model (the worker path).
    from aios.harness.loop import _resolve_capability_model
    from aios.harness.step_context import (
        compose_step_context,
        compute_step_prelude,
        prelude_overhead_local,
    )

    session = await sessions_service.get_session_basic(pool, session_id, account_id=_ACCOUNT)
    agent = await agents_service.load_for_session(pool, session, account_id=_ACCOUNT)
    capability_model = await _resolve_capability_model(pool, agent.model, account_id=_ACCOUNT)
    assert capability_model == "wf/text-only", (
        f"resolver should map workflow:<id> → wf/text-only; got {capability_model!r}"
    )
    prelude = await compute_step_prelude(
        pool,
        session_id,
        account_id=_ACCOUNT,
        session=session,
        agent=agent,
        channels=[],
        memory_store_echoes=[],
    )
    overhead = prelude_overhead_local(prelude)
    windowed = await sessions_service.read_windowed_events(
        pool,
        session_id,
        account_id=_ACCOUNT,
        window_min=agent.window_min,
        window_max=overhead.total + 5000,
        model=agent.model,
        overhead_local=overhead,
    )
    ctx = await compose_step_context(
        pool=pool,
        session=session,
        account_id=_ACCOUNT,
        agent=agent,
        channels=[],
        prelude=prelude,
        events=windowed.events,
        omission=windowed.omission,
        capability_model=capability_model,
        persist_reminders=False,
    )

    # No message in the built payload carries an image_url part — the
    # build-time pass stripped it because the resolved model is text-only.
    for msg in ctx.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                assert not (isinstance(part, dict) and part.get("type") == "image_url"), (
                    f"text-only workflow session must not receive a replayed image_url "
                    f"part; found one in role={msg.get('role')!r}"
                )
    # The tool result is present as a text marker (the part was downgraded, not dropped).
    tool_msg = next(m for m in ctx.messages if m.get("role") == "tool")
    assert isinstance(tool_msg["content"], list)
    assert any(
        "does not support image input" in p.get("text", "")
        for p in tool_msg["content"]
        if isinstance(p, dict)
    )
