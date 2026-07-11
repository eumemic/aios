"""Temporarily suppress quiescence nudges for one open obligation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aios.db import queries
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    duration_seconds: int = Field(ge=1)


async def defer_obligation_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    try:
        args = _Args.model_validate(arguments)
    except ValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    until = datetime.now(UTC) + timedelta(seconds=args.duration_seconds)
    async with pool.acquire() as conn, conn.transaction():
        open_ids = await queries.get_open_request_ids(conn, session_id, account_id=account_id)
        if args.request_id not in open_ids:
            raise ToolBail("unknown_request: request_id is not an open obligation of this session")
        deadline = await queries.get_request_deadline(
            conn, session_id, request_id=args.request_id
        )
        if deadline is not None and until > deadline:
            raise ToolBail(
                f"defer_exceeds_caller_deadline: requested until {until.isoformat()} "
                f"but caller deadline is {deadline.isoformat()}"
            )
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="lifecycle",
            data={
                "event": "request_deferred",
                "request_id": args.request_id,
                "until": until.isoformat(),
            },
        )
    return {
        "status": "deferred",
        "request_id": args.request_id,
        "deferred_until": until.isoformat(),
    }


DESCRIPTION = (
    "Snooze quiescence nudges for one obligation while leaving it open. The tool returns "
    "immediately and you remain free to work or answer early. When the duration expires, "
    "the durable sweep prompts you again. A defer cannot exceed a delegated caller deadline."
)

registry.register(
    name="defer_obligation",
    description=DESCRIPTION,
    parameters_schema=_Args.model_json_schema(),
    handler=defer_obligation_handler,
    transport="agent_tool",
)
