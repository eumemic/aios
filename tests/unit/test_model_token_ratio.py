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
        * No ``LIMIT`` window — lifetime calibration.
        """
        conn = _mock_conn([])
        await model_token_class_ratios(conn, "model-x", account_id="acc_test_stub")
        sql = conn.fetch.await_args.args[0]
        assert "local_tokens_by_class" in sql
        assert "cache_read_input_tokens" not in sql
        assert "cache_creation_input_tokens" not in sql
        assert "data ? 'local_tokens_by_class'" in sql
        assert "LIMIT" not in sql

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
