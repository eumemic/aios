"""E2E tests for :func:`aios.db.queries.model_token_ratio` against a real
Postgres with migration 0024's partial index applied.

Unlike the mock-based unit tests in ``tests/unit/test_model_token_ratio.py``
(which pin only the Python arithmetic), these cases exercise the SQL
itself — JSON extraction, the ``(data->>'is_error')::boolean`` cast, the
``data ? 'local_tokens'`` existence predicate, the ``(data->>'model') = $1``
partitioning, and the ``ORDER BY seq DESC LIMIT $2`` sliding window.

Every test uses a UUID-based synthetic model name, so spans seeded by one
test cannot contaminate another test's aggregate even though the ``events``
table persists across cases within the module-scoped pool fixture.
"""

from __future__ import annotations

import uuid

import pytest

from aios.db import queries
from aios.services import sessions as sessions_service
from tests.e2e.harness import Harness


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
            "model": model,
        },
    )


async def _seed_error_span(harness: Harness, session_id: str) -> None:
    """Insert an error-branch ``model_request_end`` — the shape emitted
    by the except-path in ``run_session_step``.  Missing ``local_tokens``
    and ``model`` keys; ``is_error = True``.  Must be excluded by the
    partial index and by the query's WHERE clause.
    """
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
    )


class TestModelTokenRatioSQL:
    async def test_below_n_returns_1(self, harness: Harness) -> None:
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(5):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            assert await queries.model_token_ratio(conn, model, n=30) == 1.0

    async def test_at_n_returns_sum_ratio(self, harness: Harness) -> None:
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_ratio(conn, model, n=30)
        assert ratio == pytest.approx(1.5)

    async def test_sums_cache_tokens_into_actual(self, harness: Harness) -> None:
        """Anthropic's usage splits input into plain + cache_read +
        cache_creation.  The ratio has to count the total prefill, not just
        the uncached slice."""
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(30):
            await _seed_valid_span(
                harness,
                session.id,
                model=model,
                local_tokens=100,
                input_tokens=50,
                cache_read=60,
                cache_creation=40,
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_ratio(conn, model, n=30)
        # total_actual per span = 50+60+40 = 150 → ratio = 150/100 = 1.5
        assert ratio == pytest.approx(1.5)

    async def test_excludes_error_spans(self, harness: Harness) -> None:
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        for _ in range(30):
            await _seed_error_span(harness, session.id)
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_ratio(conn, model, n=30)
        assert ratio == pytest.approx(1.5)

    async def test_partitions_by_model_string(self, harness: Harness) -> None:
        model_a = f"test-model-{uuid.uuid4().hex[:8]}"
        model_b = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model_a, local_tokens=100, input_tokens=150
            )
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model_b, local_tokens=100, input_tokens=120
            )
        async with harness._pool.acquire() as conn:
            ratio_a = await queries.model_token_ratio(conn, model_a, n=30)
            ratio_b = await queries.model_token_ratio(conn, model_b, n=30)
        assert ratio_a == pytest.approx(1.5)
        assert ratio_b == pytest.approx(1.2)

    async def test_cross_session_aggregation(self, harness: Harness) -> None:
        """Ratio pools spans across every session in the DB for a given
        model — a new session on a known model inherits the calibration
        from prior traffic."""
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session_a = await harness.start("seed-a")
        session_b = await harness.start("seed-b")
        for _ in range(15):
            await _seed_valid_span(
                harness, session_a.id, model=model, local_tokens=100, input_tokens=150
            )
        for _ in range(15):
            await _seed_valid_span(
                harness, session_b.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_ratio(conn, model, n=30)
        assert ratio == pytest.approx(1.5)

    async def test_sliding_window_discards_old_samples(self, harness: Harness) -> None:
        """LIMIT N in the CTE keeps only the most recent N; older samples
        with a different ratio fall out of the aggregate as traffic
        accumulates."""
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
            # n=30 → only the newer regime counts.
            assert await queries.model_token_ratio(conn, model, n=30) == pytest.approx(1.5)
            # n=60 → both regimes count:
            # total_actual = 30*200 + 30*150 = 10500; total_local = 60*100 = 6000.
            assert await queries.model_token_ratio(conn, model, n=60) == pytest.approx(1.75)

    async def test_zero_local_tokens_excluded(self, harness: Harness) -> None:
        """The WHERE clause filters ``(data->>'local_tokens')::bigint > 0``
        so zero-local rows never contribute (shouldn't happen in practice
        but must be robust to the edge)."""
        model = f"test-model-{uuid.uuid4().hex[:8]}"
        session = await harness.start("seed")
        # 30 with local_tokens=0 (all excluded).
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=0, input_tokens=999
            )
        # 30 valid.
        for _ in range(30):
            await _seed_valid_span(
                harness, session.id, model=model, local_tokens=100, input_tokens=150
            )
        async with harness._pool.acquire() as conn:
            ratio = await queries.model_token_ratio(conn, model, n=30)
        assert ratio == pytest.approx(1.5)
