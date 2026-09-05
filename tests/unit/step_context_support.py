"""Shared scaffolding for unit tests that drive ``compose_step_context``.

The composer's two service seams (the workspace-path read and the account
timezone) are stubbed; the reminder writer's ``append_event`` seam is left
to the caller so a test can mint rows into its own fake log.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

from aios.harness.step_context import StepContext, StepPrelude, compose_step_context
from aios.models.agents import AgentBinding, OutputStyle, StepSurface
from aios.models.events import Event

ACCOUNT = "acc_unit"
SESSION = "sess_unit"
FIXED_CREATED_AT = datetime(2026, 8, 25, tzinfo=UTC)


def make_step_surface(*, output_style: OutputStyle = "default") -> StepSurface:
    return StepSurface(
        model="gpt-test",
        system="you are a test agent",
        tools=[],
        skills=[],
        mcp_servers=[],
        http_servers=[],
        litellm_extra={},
        window_min=1,
        window_max=10,
        preempt_policy="wait",
        output_style=output_style,
        binding=AgentBinding(agent_id="agt_unit", version=1),
    )


def message_event(
    seq: int,
    role: str,
    content: str = "hi",
    *,
    reacting_to: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    data: dict[str, Any] = {"role": role, "content": content}
    if reacting_to is not None:
        data["reacting_to"] = reacting_to
    if metadata is not None:
        data["metadata"] = metadata
    return Event(
        id=f"evt_{seq:04d}",
        session_id=SESSION,
        seq=seq,
        kind="message",
        data=data,
        cumulative_tokens=None,
        created_at=FIXED_CREATED_AT,
    )


def make_prelude(
    *, system_prompt: str = "sys", reminders_upper_bound_local: int = 0
) -> StepPrelude:
    return StepPrelude(
        system_prompt=system_prompt,
        tools=[],
        skill_versions=[],
        obligations=[],
        reminders_upper_bound_local=reminders_upper_bound_local,
    )


class FakeSession:
    id = SESSION
    focal_channel = None


async def compose_with_stubs(
    agent: StepSurface,
    events: list[Event],
    *,
    channels: list[str] | None = None,
    persist_reminders: bool = False,
    append_event: Callable[..., Awaitable[Event]] | None = None,
) -> StepContext:
    """Run the real composer over ``events`` with its service seams stubbed.

    ``append_event`` replaces ``queries.append_event`` — the reminder writer's
    seam — for a ``persist_reminders=True`` run; ``None`` leaves the real one
    (only valid when nothing will be written).
    """
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                "aios.services.sessions.load_session_workspace_path",
                new=AsyncMock(return_value=None),
            )
        )
        stack.enter_context(
            mock.patch(
                "aios.services.accounts.resolve_effective_timezone",
                new=AsyncMock(return_value="UTC"),
            )
        )
        if append_event is not None:
            stack.enter_context(mock.patch("aios.db.queries.append_event", new=append_event))
        return await compose_step_context(
            pool=MagicMock(),
            session=FakeSession(),  # type: ignore[arg-type]
            account_id=ACCOUNT,
            agent=agent,
            channels=channels or [],
            prelude=make_prelude(),
            events=events,
            persist_image_rewrites=False,
            persist_reminders=persist_reminders,
        )
