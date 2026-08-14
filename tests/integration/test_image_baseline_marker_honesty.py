"""Baseline-marker honesty: arithmetic applied must match the marker recorded.

Migration 0161 adds ``sessions.token_baseline_v`` / ``events.token_baseline_v``
defaulting to 1, while the append path unconditionally applies the new
image-aware delta. That combination lets a session record ``v1`` while storing
``v2`` cumulative arithmetic — and, because ``cumulative_image_mass`` is forced
to 0 on a v1 row, the image mass silently vanishes from a total that already
contains it. These tests pin the two halves:

* RED: a v1-marked session must never store image-inflated cumulative_tokens.
* POSITIVE CONTROL: a genuinely v2 session still gets full image accounting.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries

# A data URI large enough that image-aware counting is unmistakable: the
# pre-fix ``litellm.token_counter`` prices an image part at a near-constant
# ~90 tokens regardless of size, so a multi-thousand-token delta can only
# come from the v2 image-aware path.
_BIG_DATA_URI = "data:image/jpeg;base64," + ("a" * 8000)

# NOTE(shape): the image message must be a TOOL-role event, not a user one.
# ``_event_token_delta`` renders a USER event via ``render_user_event``, which
# stringifies list content — so a user message's data URI is counted as plain
# TEXT under BOTH baselines (~1071 tokens either way) and shows no v1/v2
# divergence at all.  A tool-role event is passed to ``approx_tokens([data])``
# as the raw dict, which is exactly where litellm's image blindness lives
# (v1 ~96 tokens vs v2 ~1102).  Using a user message here would make this test
# pass for the wrong reason.
_IMAGE_MESSAGE: dict[str, Any] = {
    "role": "tool",
    "tool_call_id": "tc_image_1",
    "name": "screenshot",
    "content": [
        {"type": "text", "text": "screenshot"},
        {"type": "image_url", "image_url": {"url": _BIG_DATA_URI}},
    ],
}

_PLAIN_MESSAGE: dict[str, Any] = {"role": "user", "content": "hello"}


async def _seed(conn: asyncpg.Connection[Any], *, baseline_v: int) -> tuple[str, str]:
    """Seed (account, env, agent, session) and return (account_id, session_id)."""
    from aios.ids import ACCOUNT, AGENT, ENVIRONMENT, SESSION, make_id

    account_id = make_id(ACCOUNT)
    env_id = make_id(ENVIRONMENT)
    agent_id = make_id(AGENT)
    session_id = make_id(SESSION)

    await conn.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ($1, NULL, TRUE, $2)",
        account_id,
        f"acct-{account_id[-6:]}",
    )
    await conn.execute(
        "INSERT INTO environments (id, name, config, account_id) VALUES ($1, $2, '{}'::jsonb, $3)",
        env_id,
        f"env-{env_id[-6:]}",
        account_id,
    )
    await conn.execute(
        "INSERT INTO agents (id, name, model, system, tools, skills, mcp_servers, "
        "http_servers, description, metadata, litellm_extra, window_min, window_max, "
        "preempt_policy, version, account_id) "
        "VALUES ($1, $2, 'openrouter/test', '', '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, "
        "'[]'::jsonb, NULL, '{}'::jsonb, '{}'::jsonb, 50000, 150000, 'wait', 1, $3)",
        agent_id,
        f"agent-{agent_id[-6:]}",
        account_id,
    )
    await conn.execute(
        "INSERT INTO sessions (id, agent_id, environment_id, agent_version, title, "
        "metadata, workspace_volume_path, env, account_id, token_baseline_v) "
        "VALUES ($1, $2, $3, 1, NULL, '{}'::jsonb, $4, '{}'::jsonb, $5, $6)",
        session_id,
        agent_id,
        env_id,
        f"/tmp/{session_id}",
        account_id,
        baseline_v,
    )
    return account_id, session_id


async def _append(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, data: dict[str, Any]
) -> Any:
    return await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data=data,
    )


async def _row(conn: asyncpg.Connection[Any], event_id: str) -> asyncpg.Record:
    row = await conn.fetchrow(
        "SELECT cumulative_tokens, cumulative_image_mass, cumulative_text_mass, "
        "token_baseline_v FROM events WHERE id = $1",
        event_id,
    )
    assert row is not None
    return row


@pytest.mark.integration
async def test_v1_session_does_not_get_v2_image_arithmetic(
    live_conn: asyncpg.Connection[Any],
) -> None:
    """RED: a v1-marked session must not store image-inflated cumulative_tokens.

    Pre-fix this fails: the append applies the image-aware delta (thousands of
    tokens) but stamps ``token_baseline_v = 1`` and forces
    ``cumulative_image_mass = 0`` — arithmetic and marker disagree, and the
    image mass is invisible in the per-class columns while still inflating the
    total the windower drops on.
    """
    account_id, session_id = await _seed(live_conn, baseline_v=1)

    plain = await _append(live_conn, account_id, session_id, _PLAIN_MESSAGE)
    plain_row = await _row(live_conn, plain.id)
    baseline_total = plain_row["cumulative_tokens"]

    image = await _append(live_conn, account_id, session_id, _IMAGE_MESSAGE)
    row = await _row(live_conn, image.id)

    image_delta = row["cumulative_tokens"] - baseline_total

    # The marker must be honest about the arithmetic that produced the row.
    assert row["token_baseline_v"] == 1

    # v1 arithmetic prices an image part at litellm's near-constant ~90 tokens
    # (measured: 96 for this payload). A ~1100-token delta is v2 arithmetic
    # under a v1 marker.
    assert image_delta < 500, (
        f"v1 session received v2 image-aware arithmetic: delta={image_delta} "
        f"tokens while recording token_baseline_v=1"
    )

    # And the per-class columns must reconcile: whatever is in the total must
    # be attributed, never silently dropped into a zeroed image column.
    assert row["cumulative_image_mass"] == 0


@pytest.mark.integration
async def test_v2_session_gets_full_image_accounting(
    live_conn: asyncpg.Connection[Any],
) -> None:
    """POSITIVE CONTROL: a genuinely v2 session counts the image payload.

    Without this, "no mixed semantics" would pass on a build that simply
    counts nothing.
    """
    account_id, session_id = await _seed(live_conn, baseline_v=2)

    plain = await _append(live_conn, account_id, session_id, _PLAIN_MESSAGE)
    baseline_total = (await _row(live_conn, plain.id))["cumulative_tokens"]

    image = await _append(live_conn, account_id, session_id, _IMAGE_MESSAGE)
    row = await _row(live_conn, image.id)

    image_delta = row["cumulative_tokens"] - baseline_total

    assert row["token_baseline_v"] == 2
    # The 8 KB data URI must be priced at its real payload cost, not ~90.
    assert image_delta > 500, f"v2 session lost image accounting: delta={image_delta}"
    # And it must be attributed to the image class, not hidden in text.
    assert row["cumulative_image_mass"] > 500


@pytest.mark.integration
async def test_calibration_fit_admits_only_v2_lineage_spans(
    live_conn: asyncpg.Connection[Any],
) -> None:
    """The v2 calibration fit must not train on v1-lineage spans.

    The span stamp is now the SESSION's marker (``loop.py``), not a constant
    ``2``, so this filter is tied to real lineage. Pinned at the DB level:
    a v1-stamped span is invisible to the fit even when otherwise well-formed.
    """
    import json

    from aios.db.queries.events import model_token_class_ratio_fit

    account_id, session_id = await _seed(live_conn, baseline_v=1)
    model = "openrouter/lineage-test"

    async def _span(seq: int, baseline: int, local: int, provider: int) -> None:
        from aios.ids import EVENT, make_id

        await live_conn.execute(
            "INSERT INTO events (id, session_id, seq, kind, data, account_id) "
            "VALUES ($1, $2, $3, 'span', $4::jsonb, $5)",
            make_id(EVENT),
            session_id,
            seq,
            json.dumps(
                {
                    "event": "model_request_end",
                    "is_error": False,
                    "model": model,
                    "local_tokens": local,
                    "local_tokens_by_class": {"text": local},
                    "model_usage": {"input_tokens": provider},
                    "token_baseline_v": baseline,
                }
            ),
            account_id,
        )

    # 40 v1-lineage spans: well-formed in every respect EXCEPT lineage.
    for i in range(40):
        await _span(1000 + i, 1, 100, 250)

    _, n_samples = await model_token_class_ratio_fit(live_conn, model, account_id=account_id)
    assert n_samples == 0, f"v1-lineage spans leaked into the v2 fit: {n_samples} admitted"

    # Genuine v2 lineage IS admitted (positive control for the filter itself).
    for i in range(40):
        await _span(2000 + i, 2, 100, 250)

    _, n_samples_v2 = await model_token_class_ratio_fit(live_conn, model, account_id=account_id)
    assert n_samples_v2 == 40, f"expected 40 v2 spans admitted, got {n_samples_v2}"


@pytest.mark.integration
async def test_non_message_append_satisfies_image_mass_not_null(
    live_conn: asyncpg.Connection[Any],
) -> None:
    """RED: a ``span`` append must not violate the NOT NULL on image mass.

    Migration 0161 declares ``events.cumulative_image_mass`` NOT NULL DEFAULT 0
    (its five 0127 siblings are nullable). A column DEFAULT applies only when
    the column is OMITTED from an INSERT -- ``append_event`` ENUMERATES it, so
    the ``None`` it computes for every non-message kind reaches the DB and
    raises NotNullViolationError. Spans are ordinary sweep telemetry appended
    through the SAME ``append_event`` as messages, so pre-fix this fires on
    every span, lifecycle, and interrupt append.

    Payload is the exact one from the failing CI run.
    """
    account_id, session_id = await _seed(live_conn, baseline_v=1)

    event = await queries.append_event(
        live_conn,
        account_id=account_id,
        session_id=session_id,
        kind="span",
        data={"event": "sweep.batch_filter_start", "candidate_count": 1},
    )

    row = await live_conn.fetchrow(
        "SELECT kind, cumulative_image_mass, cumulative_tokens, cumulative_text_mass "
        "FROM events WHERE id = $1",
        event.id,
    )
    assert row is not None
    assert row["kind"] == "span"
    # The constraint is satisfied by the column's own declared default...
    assert row["cumulative_image_mass"] == 0
    # ...and the substitution stays confined to that one NOT NULL column: the
    # nullable siblings still record "no cumulative data" for a non-message row,
    # which is what every reader's ``cumulative_tokens IS NOT NULL`` filter uses.
    assert row["cumulative_tokens"] is None
    assert row["cumulative_text_mass"] is None


@pytest.mark.integration
async def test_every_non_message_kind_appends(
    live_conn: asyncpg.Connection[Any],
) -> None:
    """The same NOT NULL exposure applies to all non-message kinds, not just span."""
    account_id, session_id = await _seed(live_conn, baseline_v=1)

    for kind in ("span", "lifecycle", "interrupt"):
        event = await queries.append_event(
            live_conn,
            account_id=account_id,
            session_id=session_id,
            kind=kind,  # type: ignore[arg-type]
            data={"event": f"{kind}.probe"},
        )
        mass = await live_conn.fetchval(
            "SELECT cumulative_image_mass FROM events WHERE id = $1", event.id
        )
        assert mass == 0, f"{kind} append stored {mass!r} for cumulative_image_mass"
