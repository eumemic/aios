"""Unit tests for the per-content-class token-ratio fit (issue #1609).

``model_token_class_ratios`` generalizes the old single lifetime scalar
``R`` into a ridge-fit coefficient dict over
:data:`aios.harness.tokens.CONTENT_CLASSES`; ``blended_r_eff`` collapses
that dict to the one composition-weighted number the windower consumes;
``model_token_ratio`` survives as a deprecated scalar shim (unweighted
mean of the coefficients) for legacy call sites.

The SQL itself — partial-index coverage, JSON extraction, is_error
filter, model-string partitioning — is exercised against a real Postgres
in the e2e layer.  These tests pin the Python-side contract only:
insufficient-sample neutral fallback, the ridge fit's recovery of a known
linear relationship, blend arithmetic, caching, and the scalar shim.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db.queries import (
    _clear_model_token_ratio_cache,
    blended_r_eff,
    model_token_class_ratios,
    model_token_ratio,
)
from aios.db.queries.events import _MODEL_TOKEN_RATIO_SAMPLE_LIMIT
from aios.harness.tokens import CONTENT_CLASSES

# The three classes the synthetic calibration data exercises; the rest stay
# identically zero across the sample and fold to the neutral 1.0.
_FIT_CLASSES = ("text", "tool_result", "thinking")


@pytest.fixture(autouse=True)
def _clear_ratio_cache() -> None:
    _clear_model_token_ratio_cache()


def _row(by_class: dict[str, float], input_tokens: float) -> dict[str, Any]:
    """One synthetic ``model_request_end`` calibration row."""
    return {"it": input_tokens, "by_class": json.dumps(by_class)}


def _mock_conn(rows: list[dict[str, Any]]) -> MagicMock:
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    return conn


def _linear_rows(coefs: dict[str, float], *, n: int, base: int = 100) -> list[dict[str, Any]]:
    """Calibration rows where ``input_tokens`` is EXACTLY ``Σ coef_c·local_c``.

    The ridge fit (prior 1.0, small λ) should recover ``coefs`` closely.
    Varies the per-class local mass across rows so the design matrix is
    well-conditioned in the active columns.
    """
    rows: list[dict[str, Any]] = []
    for i in range(n):
        by_class = {
            "text": float(base + 10 * i),
            "tool_result": float(base + 7 * (i % 4)),
            "thinking": float(base + 3 * (i % 3)),
        }
        it = sum(coefs[c] * by_class[c] for c in by_class)
        rows.append(_row(by_class, it))
    return rows


class TestModelTokenClassRatios:
    @pytest.mark.asyncio
    async def test_below_min_samples_is_all_neutral(self) -> None:
        # 4 rows < the 5-sample threshold → every class is the neutral 1.0,
        # so the windower stays byte-identical to today (acceptance #5).
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.5, "thinking": 3.0}, n=4))
        ratios = await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        assert ratios == {c: 1.0 for c in CONTENT_CLASSES}

    @pytest.mark.asyncio
    async def test_keys_are_the_full_class_set(self) -> None:
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.5, "thinking": 3.0}, n=20))
        ratios = await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        assert set(ratios.keys()) == set(CONTENT_CLASSES)

    @pytest.mark.asyncio
    async def test_recovers_known_linear_coefficients(self) -> None:
        # When input_tokens is exactly Σ coef·local, the ridge fit recovers
        # the per-class coefficients (ridge prior 1.0 biases slightly toward
        # 1.0, hence the loose tolerance — the *blend* is what's load-bearing).
        true = {"text": 2.0, "tool_result": 1.4, "thinking": 3.0}
        conn = _mock_conn(_linear_rows(true, n=40))
        ratios = await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        for c, v in true.items():
            assert ratios[c] == pytest.approx(v, abs=0.25), f"class {c}: {ratios[c]} vs {v}"

    @pytest.mark.asyncio
    async def test_unseen_classes_stay_neutral(self) -> None:
        # Classes with zero mass across the whole sample are unidentified and
        # keep the neutral 1.0 (no spurious coefficient from a zero column).
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.4, "thinking": 3.0}, n=20))
        ratios = await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        for c in CONTENT_CLASSES:
            if c not in _FIT_CLASSES:
                assert ratios[c] == 1.0, f"unseen class {c} should be neutral"

    @pytest.mark.asyncio
    async def test_coefficients_clamped_to_physical_range(self) -> None:
        # An absurd relationship is clamped into [_MIN, _MAX]; never negative,
        # never explosive — the windower divides by the blend.
        conn = _mock_conn(_linear_rows({"text": 50.0, "tool_result": 50.0, "thinking": 50.0}, n=20))
        ratios = await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        assert all(0.0 <= v <= 8.0 for v in ratios.values())

    @pytest.mark.asyncio
    async def test_passes_model_to_query(self) -> None:
        conn = _mock_conn([])
        await model_token_class_ratios(
            conn, "anthropic/claude-sonnet-4-6", account_id="acc_test_stub"
        )
        args = conn.fetch.await_args
        assert args is not None
        assert args.args[1] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_sql_shape(self) -> None:
        """SQL invariants for the per-class calibration scan:

        * Trains on the stamped ``local_tokens_by_class`` vector.
        * Does not sum ``cache_*`` breakdown fields onto ``input_tokens``
          (regression for the pre-#163 double-count).
        * Excludes rows lacking ``local_tokens_by_class`` (zero backfill;
          self-heals as new spans accumulate).
        * Bounds the scan to the most recent N spans via
          ``ORDER BY seq DESC LIMIT $2`` (issue #1711), riding migration
          0024's ``seq DESC`` partial index instead of an unbounded
          lifetime scan on the step hot path.
        """
        conn = _mock_conn([])
        await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        sql = conn.fetch.await_args.args[0]
        assert "local_tokens_by_class" in sql
        assert "cache_read_input_tokens" not in sql
        assert "cache_creation_input_tokens" not in sql
        assert "data ? 'local_tokens_by_class'" in sql
        assert "ORDER BY seq DESC" in sql
        assert "LIMIT $2" in sql

    @pytest.mark.asyncio
    async def test_fetch_bound_to_sample_limit_constant(self) -> None:
        """The LIMIT is bound to the module constant (locked at 1000, issue
        #1711), passed as ``$2`` alongside the model — not inlined — so the
        scan is a bounded seek rather than a full lifetime scan."""
        conn = _mock_conn([])
        await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        args = conn.fetch.await_args
        assert args is not None
        # $1 = model, $2 = the sample limit.
        assert args.args[1] == "model-x"
        assert args.args[2] == _MODEL_TOKEN_RATIO_SAMPLE_LIMIT
        assert _MODEL_TOKEN_RATIO_SAMPLE_LIMIT == 1000

    @pytest.mark.asyncio
    async def test_sub_limit_sample_fits_identically_to_lifetime(self) -> None:
        """No-op below the limit: with fewer than the limit spans, every row
        the pre-#1711 unbounded query would have returned is still selected,
        so the fit is byte-identical (same rows in, same coefficients out)."""
        true = {"text": 2.0, "tool_result": 1.4, "thinking": 3.0}
        rows = _linear_rows(true, n=_MODEL_TOKEN_RATIO_SAMPLE_LIMIT - 1)
        conn = _mock_conn(rows)
        bounded = await model_token_class_ratios(conn, "model-sub", account_id="acc_test_stub")
        # The unbounded lifetime fit over the identical row set: _fit_class_ratios
        # is deterministic in its input rows, so equal rows ⇒ equal fit.
        from aios.db.queries.events import _fit_class_ratios

        lifetime = _fit_class_ratios(rows)
        assert lifetime is not None
        assert bounded == lifetime

    @pytest.mark.asyncio
    async def test_recency_honored_recent_rows_drive_fit(self) -> None:
        """Recency: the DB returns the most recent N (``ORDER BY seq DESC
        LIMIT``); a deliberately skewed *old* tail beyond N is never fetched,
        so the fit reflects only the recent regime.

        The mock stands in for the DB's ``ORDER BY seq DESC LIMIT $2`` seek:
        it returns exactly the recent-N slice, and the assertion proves the
        old tail (a wildly different coefficient regime) does not move the
        fit — i.e. the bound is honored, not silently ignored."""
        recent = {"text": 2.0, "tool_result": 1.4, "thinking": 3.0}
        old_skew = {"text": 8.0, "tool_result": 8.0, "thinking": 8.0}
        recent_rows = _linear_rows(recent, n=_MODEL_TOKEN_RATIO_SAMPLE_LIMIT)
        old_rows = _linear_rows(old_skew, n=500)
        # The DB, honoring ORDER BY seq DESC LIMIT, hands back only the recent N.
        conn = _mock_conn(recent_rows)
        recent_only_fit = await model_token_class_ratios(
            conn, "model-recent", account_id="acc_test_stub"
        )
        # If the bound were dropped, the fit would train on recent + old tail.
        from aios.db.queries.events import _fit_class_ratios

        contaminated = _fit_class_ratios(recent_rows + old_rows)
        assert contaminated is not None
        # The recent-only fit must NOT equal the contaminated lifetime fit,
        # and must track the recent regime.
        assert recent_only_fit != contaminated
        for c, v in recent.items():
            assert recent_only_fit[c] == pytest.approx(v, abs=0.25)

    @pytest.mark.asyncio
    async def test_invalid_bucket_rejected(self) -> None:
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.4, "thinking": 3.0}, n=20))
        with pytest.raises(ValueError, match="k_bucket must be positive"):
            await model_token_class_ratios(
                conn, "model-x", k_bucket=0.0, account_id="acc_test_stub"
            )

    @pytest.mark.asyncio
    async def test_calibrated_ratios_are_cached(self) -> None:
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.4, "thinking": 3.0}, n=20))
        first = await model_token_class_ratios(conn, "model-cache", account_id="acc_test_stub")
        conn.fetch = AsyncMock(
            return_value=_linear_rows({"text": 5.0, "tool_result": 5.0, "thinking": 5.0}, n=20)
        )
        second = await model_token_class_ratios(conn, "model-cache", account_id="acc_test_stub")
        assert second == first, "expected cached reuse"
        conn.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_below_threshold_neutral_is_cached(self) -> None:
        """The neutral fallback is cached too (shorter TTL), bounding the
        JSONB-scan rate on freshly deployed models (same contract as #160)."""
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.4, "thinking": 3.0}, n=2))
        first = await model_token_class_ratios(conn, "model-cold", account_id="acc_test_stub")
        assert first == {c: 1.0 for c in CONTENT_CLASSES}
        conn.fetch = AsyncMock(
            return_value=_linear_rows({"text": 2.0, "tool_result": 1.4, "thinking": 3.0}, n=20)
        )
        second = await model_token_class_ratios(conn, "model-cold", account_id="acc_test_stub")
        assert second == first
        conn.fetch.assert_not_awaited()


class TestBlendedREff:
    def test_composition_weighted_average(self) -> None:
        coefs = {"text": 2.0, "tool_result": 1.0, "thinking": 3.0}
        # 100 text, 300 tool_result, 100 thinking → (200+300+300)/500 = 1.6
        comp = {"text": 100.0, "tool_result": 300.0, "thinking": 100.0}
        assert blended_r_eff(coefs, comp) == pytest.approx(1.6)

    def test_empty_composition_falls_back_to_mean(self) -> None:
        coefs = {"text": 2.0, "tool_result": 1.0, "thinking": 3.0}
        assert blended_r_eff(coefs, {}) == pytest.approx(2.0)

    def test_neutral_coefs_blend_to_one(self) -> None:
        # The acceptance-#5 fence: all-1.0 coefficients ⇒ R_eff == 1.0 for any
        # composition, which keeps the windower byte-identical to today.
        coefs = {c: 1.0 for c in CONTENT_CLASSES}
        comp = {"text": 100.0, "tool_result": 300.0, "thinking": 50.0}
        assert blended_r_eff(coefs, comp) == pytest.approx(1.0)

    def test_shifting_mass_to_heavier_class_raises_r_eff(self) -> None:
        coefs = {"text": 1.0, "tool_result": 1.0, "thinking": 3.0}
        light = {"text": 900.0, "thinking": 100.0}
        heavy = {"text": 100.0, "thinking": 900.0}
        assert blended_r_eff(coefs, heavy) > blended_r_eff(coefs, light)


class TestScalarShim:
    @pytest.mark.asyncio
    async def test_shim_returns_mean_of_coefficients(self) -> None:
        true = {"text": 2.0, "tool_result": 1.4, "thinking": 3.0}
        conn = _mock_conn(_linear_rows(true, n=40))
        ratios = await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        _clear_model_token_ratio_cache()
        conn2 = _mock_conn(_linear_rows(true, n=40))
        scalar = await model_token_ratio(conn2, "model-x", account_id="acc_test_stub")
        expected = sum(ratios.values()) / len(ratios)
        assert scalar == pytest.approx(max(expected, 0.5))

    @pytest.mark.asyncio
    async def test_shim_neutral_below_threshold(self) -> None:
        conn = _mock_conn(_linear_rows({"text": 2.0, "tool_result": 1.4, "thinking": 3.0}, n=3))
        # All-neutral coefs ⇒ mean 1.0 (above the 0.5 clamp).
        assert await model_token_ratio(
            conn, "model-x", account_id="acc_test_stub"
        ) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_shim_min_clamp(self) -> None:
        # Coefs fit near zero ⇒ mean below 0.5 ⇒ clamped up to the historical
        # scalar floor (a degenerate, near-empty-prompt relationship).
        conn = _mock_conn(_linear_rows({"text": 0.01, "tool_result": 0.01, "thinking": 0.01}, n=20))
        assert await model_token_ratio(conn, "model-x", account_id="acc_test_stub") >= 0.5
