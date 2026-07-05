"""Integration test: the #1676 clone-policy behavioral arms.

The per-column clone policy (``clone_policy.py``) silently changed live clone
behavior relative to the old hand-enumerated lists.  This test pins the
security- and correctness-relevant arms so a present-but-wrong arm can't slip
past the completeness gate (which proves *coverage*, not *safety*):

* the #1676 LIVE drift fix — the 0127 ``cumulative_*`` class-mass columns are
  now COPIED (they were dropped on every clone, running the #1609 R_eff blend
  composition-blind);
* the authority arms — ``model`` / ``litellm_extra`` / the frozen surface are
  COPIED (never widen authority through clone; #794 lattice), and
  ``parent_run_id`` / ``origin`` are RESET (a clone must not join the run's
  sweep/cancel cascade nor read a live surface as a phantom run child);
* the usage arms — token / cost counters RESET;
* the trigger ``ingest_token_hash`` arm — a fresh hash for an external_event
  trigger clone (the UNIQUE index forbids copying), NULL for a non-external one
  (the iff CHECK forbids a stray hash).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool, register_jsonb_codec
from aios.db.queries.sessions import clone_session
from aios.ids import EVENT, GITHUB_REPOSITORY, SESSION, TRIGGER, make_id

pytestmark = pytest.mark.integration

ACCOUNT = "acc_clone_arms"


@pytest.fixture
async def pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'clone-arms')",
                ACCOUNT,
            )
        yield pool
    finally:
        await pool.close()


async def _seed_agent_env(conn: asyncpg.Connection[Any]) -> tuple[str, str]:
    agent_id = make_id("agent")
    env_id = make_id("env")
    await conn.execute(
        "INSERT INTO agents (id, account_id, name, model, system, version) "
        "VALUES ($1, $2, 'a', 'openrouter/test', '', 1)",
        agent_id,
        ACCOUNT,
    )
    await conn.execute(
        "INSERT INTO environments (id, account_id, name) VALUES ($1, $2, 'e')",
        env_id,
        ACCOUNT,
    )
    return agent_id, env_id


async def test_clone_copies_authority_and_class_mass_and_resets_run_child(
    pool: asyncpg.Pool[Any],
) -> None:
    """A frozen run-child clone keeps its surface/model pin, resets its run
    lineage + usage, and carries the 0127 class-mass counters verbatim."""
    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)
        await conn.execute(
            "INSERT INTO wf_runs (id, account_id, script, script_sha, environment_id, "
            "host_semantics_epoch) VALUES ('run_p', $1, 's', 'sha', $2, 0)",
            ACCOUNT,
            env_id,
        )
        parent = make_id(SESSION)
        await conn.execute(
            """
            INSERT INTO sessions (
                id, agent_id, environment_id, agent_version, title, metadata,
                workspace_volume_path, env, focal_channel, focal_locked, account_id,
                last_event_seq, input_tokens, output_tokens, cost_microusd,
                parent_run_id, origin, surface_frozen, model, litellm_extra,
                tools, mcp_servers, http_servers, snapshot_ref, outbound_suppression)
            VALUES ($1, $2, $3, 1, 't', '{}'::jsonb, '/w/p', '{}'::jsonb, 'tg/c1',
                TRUE, $4, 2, 100, 50, 4242, 'run_p', 'background', TRUE,
                'anthropic/claude', '{"api_base":"x"}'::jsonb, '[]'::jsonb, '[]'::jsonb,
                '[]'::jsonb, 'snap_p', 'on')
            """,
            parent,
            agent_id,
            env_id,
            ACCOUNT,
        )
        for seq in (1, 2):
            await conn.execute(
                """
                INSERT INTO events (id, session_id, seq, kind, data, account_id,
                    cumulative_tokens, cumulative_messages, cumulative_text_mass,
                    cumulative_tool_result_mass, cumulative_thinking_mass,
                    cumulative_tool_use_mass, role)
                VALUES ($1, $2, $3, 'message', '{}'::jsonb, $4, $5, $6, $7, $8, $9, $10, 'user')
                """,
                make_id(EVENT),
                parent,
                seq,
                ACCOUNT,
                seq * 10,
                seq,
                seq * 3,
                seq * 4,
                seq * 5,
                seq * 6,
            )

        clone = await clone_session(
            conn, parent, account_id=ACCOUNT, workspace_path="/w/clone"
        )
        srow = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", clone.id)
        assert srow is not None

        # authority / isolation arms — COPIED (never widen through clone)
        assert srow["model"] == "anthropic/claude"
        assert srow["litellm_extra"] == {"api_base": "x"}
        assert srow["surface_frozen"] is True
        assert srow["focal_locked"] is True
        assert srow["focal_channel"] == "tg/c1"
        assert srow["outbound_suppression"] == "on"

        # run-lineage arms — RESET (no phantom run-child membership)
        assert srow["parent_run_id"] is None
        assert srow["origin"] == "foreground"

        # usage arms — RESET (paid on the parent)
        assert srow["input_tokens"] == 0
        assert srow["output_tokens"] == 0
        assert srow["cost_microusd"] == 0

        # snapshot pointer — RESET (belongs to the parent's workspace)
        assert srow["snapshot_ref"] is None
        assert srow["workspace_volume_path"] == "/w/clone"

        # the #1676 LIVE drift fix — the 0127 class-mass columns COPIED verbatim
        erows = await conn.fetch(
            "SELECT * FROM events WHERE session_id = $1 ORDER BY seq", clone.id
        )
        assert len(erows) == 2
        tail = erows[1]
        assert tail["cumulative_tokens"] == 20
        assert tail["cumulative_messages"] == 2
        assert tail["cumulative_text_mass"] == 6
        assert tail["cumulative_tool_result_mass"] == 8
        assert tail["cumulative_thinking_mass"] == 10
        assert tail["cumulative_tool_use_mass"] == 12


async def test_clone_triggers_ingest_token_arm(pool: asyncpg.Pool[Any]) -> None:
    """external_event trigger clones get a FRESH ingest hash (UNIQUE index),
    non-external ones NULL (the iff CHECK) — the source-conditional arm."""
    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)
        parent = make_id(SESSION)
        await conn.execute(
            """
            INSERT INTO sessions (id, agent_id, environment_id, agent_version, title,
                metadata, workspace_volume_path, env, account_id, last_event_seq)
            VALUES ($1, $2, $3, 1, 't', '{}'::jsonb, '/w/p', '{}'::jsonb, $4, 0)
            """,
            parent,
            agent_id,
            env_id,
            ACCOUNT,
        )
        await conn.execute(
            """
            INSERT INTO triggers (id, owner_session_id, account_id, name, source,
                source_spec, action, enabled, next_fire, consecutive_failures, running_since)
            VALUES ($1, $2, $3, 'cron1', 'cron', '{"schedule":"* * * * *"}'::jsonb,
                '{"kind":"wake_owner","content":"hi"}'::jsonb, TRUE, now(), 5, now())
            """,
            make_id(TRIGGER),
            parent,
            ACCOUNT,
        )
        await conn.execute(
            """
            INSERT INTO triggers (id, owner_session_id, account_id, name, source,
                source_spec, action, enabled, ingest_token_hash)
            VALUES ($1, $2, $3, 'ext1', 'external_event', '{}'::jsonb,
                '{"kind":"wake_owner","content":"hi"}'::jsonb, TRUE, 'parenthash')
            """,
            make_id(TRIGGER),
            parent,
            ACCOUNT,
        )

        clone = await clone_session(
            conn, parent, account_id=ACCOUNT, workspace_path="/w/clone"
        )
        trows = await conn.fetch(
            "SELECT * FROM triggers WHERE owner_session_id = $1 ORDER BY name", clone.id
        )
        by = {r["name"]: r for r in trows}
        assert len(trows) == 2

        # cron trigger: no ingest hash, runtime state reset
        assert by["cron1"]["ingest_token_hash"] is None
        assert by["cron1"]["running_since"] is None
        assert by["cron1"]["consecutive_failures"] == 0
        assert by["cron1"]["id"].startswith(f"{TRIGGER}_")

        # external_event trigger: a FRESH hash, never the parent's
        assert by["ext1"]["ingest_token_hash"] is not None
        assert by["ext1"]["ingest_token_hash"] != "parenthash"


async def test_clone_mints_fresh_github_repo_id(pool: asyncpg.Pool[Any]) -> None:
    """The github-repo global PK is minted fresh; the row is otherwise copied."""
    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)
        parent = make_id(SESSION)
        await conn.execute(
            """
            INSERT INTO sessions (id, agent_id, environment_id, agent_version, title,
                metadata, workspace_volume_path, env, account_id, last_event_seq)
            VALUES ($1, $2, $3, 1, 't', '{}'::jsonb, '/w/p', '{}'::jsonb, $4, 0)
            """,
            parent,
            agent_id,
            env_id,
            ACCOUNT,
        )
        gh_id = make_id(GITHUB_REPOSITORY)
        await conn.execute(
            """
            INSERT INTO session_github_repositories (id, session_id, rank, repo_url,
                mount_path, ciphertext, nonce, account_id)
            VALUES ($1, $2, 0, 'http://r', '/m', $3, $4, $5)
            """,
            gh_id,
            parent,
            b"cipher",
            b"nonce",
            ACCOUNT,
        )

        clone = await clone_session(
            conn, parent, account_id=ACCOUNT, workspace_path="/w/clone"
        )
        grows = await conn.fetch(
            "SELECT * FROM session_github_repositories WHERE session_id = $1", clone.id
        )
        assert len(grows) == 1
        assert grows[0]["id"] != gh_id
        assert grows[0]["id"].startswith(f"{GITHUB_REPOSITORY}_")
        assert grows[0]["repo_url"] == "http://r"
        assert grows[0]["ciphertext"] == b"cipher"
