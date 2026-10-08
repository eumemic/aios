"""The replay run tools (#2475): an operator eval workflow reads an agent's past requests.

``sample_requests`` returns a seeded sample of an agent's answered requests as refs, and
``get_request`` rebuilds one ref inline for a target model. Both run on the worker like
any run tool (:mod:`aios.workflows.run_tools`), behind ``gate_run_tool``: the run must act
for the operator and declare the tool, which no agent can hold (#794: operator principal
AND a declared capability, never ambient).

A ref a run may resolve (:func:`request_ref_granted`) is the one it was created with
(``WfRun.request_ref``, #2474) or one its own ``sample_requests`` call returned. That
second grant is read from the run's journal and keyed on the call that minted the value,
never on its shape.

``get_request`` journals a full request in its ``call_result``. The run is invisible to
agents (C1), and the journal is bounded by the run's retention.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import asyncpg
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.db.queries import requests as request_queries
from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.workflows import RequestRef, WfRun
from aios.services.requests import Missing, Rebuilt, rebuild_request

log = get_logger("aios.workflows.run_replay")

# The tools whose results grant the refs they return. Closed: a grant is keyed on the
# minting call, so adding a tool here is a deliberate authority decision.
MINTING_TOOLS: tuple[str, ...] = ("sample_requests",)

# Bounds on one sample: the span read is per agent session over the range.
MAX_SAMPLE_SIZE = 500
MAX_SAMPLE_RANGE = timedelta(days=31)

_WORKFLOW_MODEL_PREFIX = "workflow:"


def _is_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"session_id", "request_id"}
        and all(isinstance(v, str) for v in value.values())
    )


async def request_ref_granted(conn: asyncpg.Connection[Any], run: WfRun, ref: Any) -> bool:
    """Whether ``run`` may resolve ``ref``: the ref it was created with, or one its own
    ``sample_requests`` call returned. A ref-shaped value anywhere else (plain input, a
    gate resume, another tool's or call's result) grants nothing."""
    if run.request_ref is not None and ref == run.request_ref.model_dump():
        return True
    if not _is_ref(ref):
        return False
    return await wf_queries.request_ref_minted(
        conn, run.id, ref, account_id=run.account_id, minting_tools=list(MINTING_TOOLS)
    )


def captured_params_for(
    rebuilt: Rebuilt, model: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The captured params a request rendered for ``model`` keeps, and the launcher
    params that vouch for their endpoint (#823), as ``(params, launcher_extra)``.

    The params belong to the model the request was sent to, and their launcher already
    passed #823 for them, so that model keeps them, endpoint included. A request sent
    to a ``workflow:`` binding handed those same params to the bound run, so any model
    keeps them, but nothing vouches for an ``api_base`` in them: it passes only the
    allowlist, as in an inline call. Any other model gets none, so the request can't
    reach that model's endpoint with another's key."""
    captured = rebuilt.request["params"]
    if model == rebuilt.record["model"]:
        return captured, captured
    if rebuilt.record["model"].startswith(_WORKFLOW_MODEL_PREFIX):
        return captured, None
    return None, None


class _SampleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    start: AwareDatetime
    end: AwareDatetime
    n: int = Field(ge=1, le=MAX_SAMPLE_SIZE)
    seed: str | int
    cluster_cap: int = Field(default=2, ge=1)


class _GetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_ref: RequestRef
    model: str | None = None


async def invoke_replay_tool(*, run: WfRun, tool_name: str, args: Any) -> dict[str, Any]:
    """Run one replay tool. ``gate_run_tool`` already admitted the call. Returns a value,
    never raises: a bad argument or an unavailable request is an ``{"error": …}`` value."""
    try:
        if tool_name == "sample_requests":
            return await _sample_requests(run, _SampleArgs.model_validate(args))
        return await _get_request(run, _GetArgs.model_validate(args))
    except PydanticValidationError as exc:
        return {"error": f"{tool_name}: bad arguments: {exc.errors(include_url=False)}"}


async def _sample_requests(run: WfRun, args: _SampleArgs) -> dict[str, Any]:
    """A seeded sample of the answered requests ``agent_id`` sent in ``[start, end)``:
    one per payload (a clone's copy of a span is skipped; the original wins), at most
    ``cluster_cap`` per session per UTC day, ordered by a hash of ``seed`` and cut at
    ``n``. ``auto_review`` checker calls and sessions an eval run spawned are skipped.

    Each item is a ref plus what an eval needs to sample and compare: the session, when
    it was sent, the model and capability model, the agent version, the assistant message
    that answered it (``response_event_id``), and whether a blob it needs is ``missing``.
    Whether it rebuilds exactly comes back when it is used (``call_llm``, ``get_request``).

    Deterministic only over a closed past range: a request added inside the range, or a
    session deleted from it, changes the sample.
    """
    if args.start >= args.end:
        return {"error": "sample_requests: start must be before end"}
    if args.end - args.start > MAX_SAMPLE_RANGE:
        return {"error": f"sample_requests: the range is at most {MAX_SAMPLE_RANGE.days} days"}
    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        rows = await request_queries.sample_request_spans(
            conn,
            account_id=run.account_id,
            agent_id=args.agent_id,
            start=args.start,
            end=args.end,
            seed=str(args.seed),
            cluster_cap=args.cluster_cap,
            n=args.n,
        )
        shas = sorted(
            {
                row["record"][key]
                for row in rows
                for key in ("system_sha", "tools_sha", "params_sha")
            }
        )
        present = await request_queries.present_blob_shas(
            conn, account_id=run.account_id, shas=shas
        )
    return {
        "items": [
            {
                "request_ref": {"session_id": row["session_id"], "request_id": row["id"]},
                "session_id": row["session_id"],
                "created_at": _iso(row["created_at"]),
                "model": row["record"]["model"],
                "capability_model": row["record"]["capability_model"],
                "agent_version": row["record"]["binding"].get("version"),
                "response_event_id": row["response_event_id"],
                "missing": any(
                    row["record"][key] not in present
                    for key in ("system_sha", "tools_sha", "params_sha")
                ),
            }
            for row in rows
        ]
    }


async def _get_request(run: WfRun, args: _GetArgs) -> dict[str, Any]:
    """Rebuild one ref inline, rendered for ``model`` (default: the model the request
    was composed for), from the same slate the session sent: no re-windowing. ``params``
    are what :func:`captured_params_for` keeps for ``model``, as for a by-ref
    ``call_llm``; a captured request never carries an inline ``api_key``.
    """
    ref = args.request_ref
    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        granted = await request_ref_granted(conn, run, ref.model_dump())
    if not granted:
        return {
            "error": "get_request can read only a request this run was given or sampled",
            "error_kind": "request_ref_not_granted",
        }
    if args.model is not None and args.model.startswith(_WORKFLOW_MODEL_PREFIX):
        return {"error": f"get_request renders for a model, not a workflow ({args.model!r})"}
    try:
        rebuilt = await rebuild_request(
            pool,
            account_id=run.account_id,
            session_id=ref.session_id,
            request_event_id=ref.request_id,
            target_model=args.model,
        )
    except NotFoundError as exc:
        return _unavailable(str(exc))
    except Exception as exc:
        log.warning("get_request.rebuild_failed", run_id=run.id, error=str(exc))
        return _unavailable(f"the rebuild failed: {type(exc).__name__}: {exc}")
    if isinstance(rebuilt, Missing):
        return _unavailable(f"a {rebuilt.what} it needs is gone")
    model = args.model or rebuilt.record["capability_model"]
    return {
        "messages": rebuilt.request["messages"],
        "tools": rebuilt.request["tools"],
        "params": captured_params_for(rebuilt, model)[0],
        "fidelity": rebuilt.fidelity,
    }


def _unavailable(why: str) -> dict[str, Any]:
    return {
        "error": f"get_request: the request is unavailable: {why}",
        "error_kind": "request_unavailable",
    }


def _iso(value: datetime) -> str:
    return value.isoformat()
