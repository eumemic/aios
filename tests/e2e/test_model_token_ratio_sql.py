"""E2E tests for :func:`aios.db.queries.model_token_class_ratios` against a real
Postgres with migration 0024's partial index applied.

Unlike the mock-based unit tests in ``tests/unit/test_model_token_ratio.py``
(which pin only the Python arithmetic), these cases exercise the SQL
itself — JSON extraction, the ``(data->>'is_error')::boolean`` cast, the
``data ? 'local_tokens'`` existence predicate, the ``(data->>'model') = $1``
partitioning, lifetime aggregation, and standard-error bucketing.

Every test uses a UUID-based synthetic model name, so spans seeded by one
test cannot contaminate another test's aggregate even though the ``events``
table persists across cases within the module-scoped pool fixture.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import asyncpg
import pytest

from aios.db import queries
from aios.db.queries.events import _MODEL_TOKEN_RATIO_SAMPLE_LIMIT
from aios.harness.tokens import CONTENT_CLASSES
from aios.services import sessions as sessions_service
from tests.e2e.harness import Harness


def _mean_ratio(ratios: dict[str, float]) -> float:
    """Return the unweighted coefficient mean for uniform compositions."""
    return sum(ratios.values()) / len(ratios)


def _by_class(local_tokens: int) -> dict[str, float]:
    """Spread a model-neutral ``local_tokens`` total evenly across every
    content class, as a stand-in for the ``local_tokens_by_class`` vector that
    ``harness/loop.py`` stamps.

    This is a *test convenience*, not the production contract: ``loop.py``
    counts ``local_tokens`` and ``local_tokens_by_class`` with two separate
    ``litellm`` calls, so in production the per-class slices each carry their
    own framing overhead and ``sum(by_class.values()) >= local_tokens`` (the
    implementation deliberately does NOT enforce equality). The even spread
    used here happens to sum back to ``local_tokens`` only so the class-ratio reader's unweighted coefficient mean reproduces ``input/local`` for these SQL-path
    assertions; the per-class ridge fit (#1609) does not depend on that.
    """
    per = local_tokens / len(CONTENT_CLASSES)
    return {c: per for c in CONTENT_CLASSES}


async def _seed_valid_span(
    harness: Harness,
    session_id: str,
    *,
    model: str,
    local_tokens: int,
    input_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> None:
    """Insert a successful ``model_request_end`` span of the shape
    ``harness/loop.py`` stamps.  Pool-acquiring via the service wrapper
    matches the production write path.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    await sessions_service.append_event(
        harness._pool,
        session_id,
        "span",
        {
            "event": "model_request_end",
            "is_error": False,
            "model_usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
            "cost_usd": None,
            "local_tokens": local_tokens,
            "local_tokens_by_class": _by_class(local_tokens),
            "model": model,
        },
        account_id=account_id,
    )


async def _seed_error_span(harness: Harness, session_id: str) -> None:
    """Insert an error-branch ``model_request_end`` — the shape emitted
    by the except-path in ``run_session_step``.  Missing ``local_tokens``
    and ``model`` keys; ``is_error = True``.  Must be excluded by the
    partial index and by the query's WHERE clause.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    await sessions_service.append_event(
        harness._pool,
        session_id,
        "span",
        {
            "event": "model_request_end",
            "is_error": True,
            "model_usage": {},
            "cost_usd": None,
        },
        account_id=account_id,
    )


class TestModelTokenRatioSQL:
    async def test_below_min_samples_returns_1(self, harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(4):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            assert set(
                (
                    await queries.model_token_class_ratios(conn, model, account_id=account_id)
                ).values()
            ) == {1.0}

    async def test_min_samples_returns_mean_ratio(self, harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(conn, model, account_id=account_id)
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.005)

    async def test_ignores_cache_breakdown_fields(self, harness: Harness) -> None:
        """LiteLLM normalizes Anthropic's usage to the OpenAI convention:
        ``input_tokens`` is already the full prompt count (including any
        cached-read and cache-creation portions).  ``cache_read_input_tokens``
        and ``cache_creation_input_tokens`` are breakdown metrics within
        that total — they MUST NOT be summed on top.  This case pins the
        invariant: seeding spans with nonzero cache_* fields does not
        change the ratio; only ``input_tokens`` and ``local_tokens``
        contribute."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(5):
            await _seed_valid_span(
                harness,
                session.id,
                model=model,
                local_tokens=100,
                input_tokens=150,
                cache_read=60,
                cache_creation=40,
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(conn, model, account_id=account_id)
        # per-span ratio = input_tokens/local_tokens = 150/100 (cache_* ignored).
        # ratio = 150/100 = 1.5.
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.005)

    async def test_excludes_error_spans(self, harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        for _ in range(30):
            await _seed_error_span(harness, session.id)
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(conn, model, account_id=account_id)
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.005)

    async def test_partitions_by_model_string(self, harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model_a = f"test-model-{uuid.uuid4().hex[:8]}"
        model_b = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model_a, local_tokens=100, input_tokens=150
            )
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model_b, local_tokens=100, input_tokens=120
            )
        async with harness._pool.acquire() as conn:
            ratio_a = await queries.model_token_class_ratios(conn, model_a, account_id=account_id)
            ratio_b = await queries.model_token_class_ratios(conn, model_b, account_id=account_id)
        assert _mean_ratio(ratio_a) == pytest.approx(1.5, abs=0.005)
        assert _mean_ratio(ratio_b) == pytest.approx(1.2, abs=0.005)

    async def test_cross_session_aggregation(self, harness: Harness) -> None:
        """Ratio pools spans across every session in the DB for a given
        model — a new session on a known model inherits the calibration
        from prior traffic."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session_a = await harness.start("seed-a")
        session_b = await harness.start("seed-b")
        for _ in range(2):
            await _seed_valid_span(
                harness, session_a.id, model=model, local_tokens=100, input_tokens=150
            )
        for _ in range(3):
            await _seed_valid_span(
                harness, session_b.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(conn, model, account_id=account_id)
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.005)

    async def test_ridge_fit_is_magnitude_weighted(self, harness: Harness) -> None:
        """The per-class ridge fit (#1609) is a least-squares regression of
        provider ``input_tokens`` on the stamped ``local_tokens_by_class``
        vector, so it is inherently magnitude-weighted: a span carrying
        two orders of magnitude more local tokens dominates the normal
        equations.  This is the deliberate replacement for the old
        unweighted per-span-ratio mean — large, information-rich windows
        (the ones that actually risk the 1M provider cap) drive the
        calibration."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(4):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=100
            )
        await _seed_valid_span(
            harness, session.id, model=model, local_tokens=10_000, input_tokens=20_000
        )

        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(
                conn, model, k_bucket=0.001, account_id=account_id
            )

        # The 10k-local span (ratio 2.0) swamps the four 100-local spans
        # (ratio 1.0) in the least-squares fit, so the coefficient — and
        # hence the coefficients' mean — sits at ~2.0, NOT the old
        # unweighted 1.2.
        assert _mean_ratio(ratio) == pytest.approx(2.0, abs=0.01)

    async def test_lifetime_aggregate_retains_old_samples(self, harness: Harness) -> None:
        """Below the sample limit, all historical calibration samples for the
        model contribute to the aggregate (60 spans << the 1000-span bound of
        issue #1711, so the ORDER BY seq DESC LIMIT is a no-op here); older
        samples do not fall out of the window."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        # Older regime: ratio 2.0.
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=200
            )
        # Newer regime: ratio 1.5.
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            # Use a narrow bucket here to isolate lifetime aggregation:
            # mean_ratio = mean([2.0] * 30 + [1.5] * 30) = 1.75.
            ratios = await queries.model_token_class_ratios(
                conn, model, k_bucket=0.001, account_id=account_id
            )
            assert _mean_ratio(ratios) == pytest.approx(1.75)

    async def test_bucket_shrinks_as_sample_count_grows(self, harness: Harness) -> None:
        """With the same underlying ratio distribution, standard-error
        bucketing gets narrower as n grows."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for input_tokens in (140, 160, 140, 160, 140):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=input_tokens
            )
        async with harness._pool.acquire() as conn:
            coarse_ratio = await queries.model_token_class_ratios(
                conn, model, account_id=account_id
            )

        for _ in range(3):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=140
            )
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=160
            )

        queries._clear_model_token_ratio_cache()
        async with harness._pool.acquire() as conn:
            tighter_ratio = await queries.model_token_class_ratios(
                conn, model, account_id=account_id
            )

        assert abs(_mean_ratio(tighter_ratio) - 1.5) < abs(_mean_ratio(coarse_ratio) - 1.5)

    async def test_zero_local_tokens_excluded(self, harness: Harness) -> None:
        """The WHERE clause filters ``(data->>'local_tokens')::bigint > 0``
        so zero-local rows never contribute (shouldn't happen in practice
        but must be robust to the edge)."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        # 30 with local_tokens=0 (all excluded).
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=0, input_tokens=999
            )
        # 5 valid.
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(conn, model, account_id=account_id)
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.005)

    async def test_zero_input_usage_and_decimal_values_are_excluded(self, harness: Harness) -> None:
        """Real-Postgres discriminator: zero usage is poison, while a future
        non-integral value must be ignored rather than crashing the cast."""
        account_id = "acc_test_stub"
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(20):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100_000, input_tokens=0
            )
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        # JSON permits this future hostile shape despite today's int-typed provider model.
        await _seed_valid_span(
            harness,
            session.id,
            model=model,
            local_tokens=100,
            input_tokens=150.5,  # type: ignore[arg-type]
        )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(conn, model, account_id=account_id)
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.01)

    async def test_recency_bound_honors_recent_spans(self, harness: Harness) -> None:
        """Recency honored (issue #1711): seed MORE than the sample limit,
        with a deliberately skewed OLD tail (ratio 4.0) below the newest
        ``_MODEL_TOKEN_RATIO_SAMPLE_LIMIT`` spans (ratio 1.5).  The
        ``ORDER BY seq DESC LIMIT $2`` bound must select only the recent
        window, so the fit reflects the recent regime, not the old tail."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        # Old, skewed tail (ratio 4.0): seeded FIRST, so lowest seq.
        for _ in range(20):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=400
            )
        # Recent regime (ratio 1.5): fills the entire bounded window.
        for _ in range(_MODEL_TOKEN_RATIO_SAMPLE_LIMIT):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_class_ratios(
                conn, model, k_bucket=0.001, account_id=account_id
            )
        # Only the recent (1.5) spans are in-window; the 4.0 tail is excluded
        # by the LIMIT, so the fit lands at the recent regime, not a blend.
        assert _mean_ratio(ratio) == pytest.approx(1.5, abs=0.02)

    async def test_recency_fetch_is_a_bounded_index_scan(self, harness: Harness) -> None:
        """Perf guard (issue #1711 / #1798): the ``ORDER BY created_at DESC,
        session_id DESC, seq DESC LIMIT`` recency fetch must be a bounded index
        seek, not a full-model scan + Sort.

        The query orders by global recency; migration 0024's
        ``((data->>'model'), seq DESC)`` index can seek the model but cannot
        satisfy that order, so without a matching covering index the planner
        must read *every* lifetime row for the model and Sort it before the
        ``LIMIT`` applies — the limit stops bounding the scan and the cost
        grows without bound as calibration history accumulates.  Migration
        0142 adds ``events_model_calibration_recency_idx`` keyed
        ``((data->>'model'), created_at DESC, session_id DESC, seq DESC)`` so
        the plan is an ordered index scan whose ``LIMIT`` bounds the read.

        This EXPLAINs the *actual* SQL the query issues (captured, not
        duplicated) so a future ORDER-BY change that the index no longer
        covers reintroduces a ``Sort`` node and fails here.
        """
        account_id = "acc_test_stub"
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )

        class _CapturingConn:
            """Delegates to a real connection while recording the one SQL the
            ratio query issues, so the plan is checked against the live query."""

            def __init__(self, real: asyncpg.Connection[Any]) -> None:
                self._real = real
                self.sql: str | None = None
                self.args: tuple[Any, ...] = ()

            async def fetch(self, query: str, *args: Any, **kwargs: Any) -> list[asyncpg.Record]:
                self.sql = query
                self.args = args
                rows: list[asyncpg.Record] = await self._real.fetch(query, *args, **kwargs)
                return rows

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

        async with harness._pool.acquire() as conn:
            capturing = _CapturingConn(conn)
            # Fresh model name ⇒ cache miss ⇒ a real DB fetch we can capture.
            await queries.model_token_class_ratios(
                cast("asyncpg.Connection[Any]", capturing), model, account_id=account_id
            )
            assert capturing.sql is not None, "the ratio query issued no fetch to EXPLAIN"

            await conn.execute("ANALYZE events")
            # Nudge the planner off a whole-table seq scan so the plan reveals
            # whether an ordered index path exists that satisfies the ORDER BY
            # without a Sort.  If none does (the regression), a Sort node
            # survives and the assertion below fails.
            async with conn.transaction():
                await conn.execute("SET LOCAL enable_seqscan = off")
                plan_rows = await conn.fetch(f"EXPLAIN {capturing.sql}", *capturing.args)
            plan = "\n".join(row[0] for row in plan_rows)

        assert "events_model_calibration_recency_idx" in plan, (
            f"recency fetch did not use the covering index:\n{plan}"
        )
        assert "Sort" not in plan, (
            f"recency fetch still sorts — the LIMIT no longer bounds the scan:\n{plan}"
        )
