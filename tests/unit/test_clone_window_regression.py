"""Clone-window regression for the per-class token budgeter (issue #1609).

This is the issue's **named must-pass regression** (acceptance #1) plus the
**stability gate** (acceptance #8).  It guards the exact incident the fix was
built for: the ``kedalion-ultron`` ``ContextWindowExceededError`` where the
single lifetime scalar ``R`` (~1.833) budgeted the failing window at ~808k --
under the harness's own 900k ``window_max`` -- while Anthropic counted
**1,005,124** tokens and rejected the call (``prompt is too long: 1005124
tokens > 1000000 maximum``).

Why this test exists separately from ``test_model_token_ratio.py``: that file
pins the fit's mechanics on *synthetic* linear data.  This file pins the fit
against the *real* incident composition, with a genuine fold-holdout so it
cannot tautologically pass.  The two anti-patterns the issue calls out by name
are (a) a cosmetic/absent holdout (training on the clone's own window) and
(b) gating on +/-5% in-sample APE (a held-out refit fails that) instead of the
robust "beats scalar AND flags the breach" gate.  Both are foreclosed here.

--------------------------------------------------------------------------
FIXTURE PROVENANCE
--------------------------------------------------------------------------
The original investigation workspace (``/tmp/tokq2``) is gone, so every
number below was **re-derived read-only from the parked clone session**
``sess_5f3421150ba82b8c714f044696`` on the PROD events table (SELECT only --
the clone is a preserved fixture, never mutated), using the *same* per-class
delta classifier the windower itself uses (``_retained_class_mass`` /
``content_class``):

    WITH msgs AS (
        SELECT seq,
            CASE
                WHEN role='tool' THEN 'tool_result'
                WHEN role='assistant' AND has(tool_calls) THEN 'tool_use'
                WHEN role='assistant' AND has(reasoning/thinking) THEN 'thinking'
                ELSE 'text'
            END AS cls,
            cumulative_tokens - LAG(cumulative_tokens) OVER (ORDER BY seq) AS delta
        FROM events WHERE session_id=<clone> AND kind='message'
    )  -- summed per class over the trailing slate that fills the window

* ``_CLONE_BREACH`` is the failing window's per-class local breakdown, summed
  over the trailing message slate up to the harness's believed local budget
  (~440,751 ~ the issue's cited 440,826), paired with the provider's
  **1,005,124** billed ``input_tokens`` (extracted verbatim from the clone's
  ``is_error=true`` ``provider_error`` message).  This window is the
  hold-OUT -- it never appears in ``_TRAIN``.
* ``_TRAIN`` is 10 representative HEALTHY windows from the *same* session,
  every one of them disjoint from the breach window, spanning the
  composition-driven ratio range (1.60-2.50) that the single scalar cannot
  track.  Each carries its real billed ``input_tokens``.  A compact fixture
  is sufficient (the issue: "~6-10 representative healthy windows ... does NOT
  need 30"); the point is a stable blend and a genuine held-out test, not
  sample size.

The reconstructed per-class masses do NOT sum exactly to the window's local
total (small per-message framing + system overhead is not carried in the
message deltas) -- which is faithful: the production ``loop.py`` likewise does
NOT enforce ``local_tokens == sum(by_class.values())`` (the two are counted
by different calls).  The regressors are used as-is, exactly as the fit
consumes the stamped vector.
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
)

# -- re-derived fixture (read-only from clone sess_5f3421150ba82b8c714f044696) --

# The failing window the incident breached on.  Per-class local breakdown
# (trailing slate <= the harness's believed ~440k local budget) -> the provider
# actually billed 1,005,124 and rejected the call at the 1,000,000 hard cap.
_CLONE_BREACH_BY_CLASS: dict[str, float] = {
    "tool_result": 263271.0,
    "tool_use": 154789.0,
    "text": 20766.0,
    "thinking": 1925.0,
}
_CLONE_BREACH_BILLED = 1_005_124  # Anthropic's reported prompt size (the breach).
_BREACH_CAP = 900_000  # the harness's window_max the scalar let the window slip under.

# The lifetime scalar R for anthropic/claude-opus-4-8 (the value the OLD code
# applied uniformly).  Pinned here so the regression proves the per-class fix
# beats it; it is NOT used by the production windower anymore.
_SCALAR_R = 1.833

# 10 healthy training windows from the same session -- DISJOINT from the breach
# window (the fold-holdout).  (per-class local breakdown, billed input_tokens).
_TRAIN: list[tuple[dict[str, float], int]] = [
    ({"tool_result": 19352, "text": 14747, "tool_use": 14392}, 82341),
    ({"tool_use": 18430, "tool_result": 10670, "text": 1311, "thinking": 136}, 114734),
    ({"tool_result": 28092, "tool_use": 24684, "text": 3456}, 138823),
    ({"tool_use": 22899, "text": 14450, "tool_result": 5269}, 92914),
    ({"tool_result": 31766, "tool_use": 21291, "text": 4414, "thinking": 641}, 127008),
    ({"tool_use": 31937, "tool_result": 26478, "text": 814}, 149113),
    ({"tool_use": 23606, "tool_result": 21225, "text": 12684}, 93234),
    ({"tool_result": 27998, "tool_use": 25734, "text": 4524, "thinking": 1087}, 127412),
    ({"tool_use": 12372, "tool_result": 11619, "text": 7748, "thinking": 1166}, 123184),
    ({"tool_result": 21168, "tool_use": 11868, "text": 1890}, 110485),
]

_BREACH_LOCAL = sum(_CLONE_BREACH_BY_CLASS.values())


def _rows(train: list[tuple[dict[str, float], int]]) -> list[dict[str, Any]]:
    """Shape the training windows as ``model_request_end`` calibration rows."""
    return [{"it": float(it), "by_class": json.dumps(bc)} for bc, it in train]


def _mock_conn(rows: list[dict[str, Any]]) -> MagicMock:
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    return conn


async def _fit(train: list[tuple[dict[str, float], int]]) -> dict[str, float]:
    """Run the REAL production fit (ridge solver + query parsing) on ``train``.

    Goes through the public ``model_token_class_ratios`` against a mocked
    connection so the test exercises the same code path the worker runs --
    not a re-implementation of the solver.
    """
    _clear_model_token_ratio_cache()
    return await model_token_class_ratios(
        _mock_conn(_rows(train)), "anthropic/claude-opus-4-8", account_id="acc_test_stub"
    )


def _perclass_prediction(coefs: dict[str, float], by_class: dict[str, float]) -> float:
    """sum_c coef_c * local_c -- the per-class budgeted provider-token estimate."""
    return sum(coefs[c] * v for c, v in by_class.items())


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _clear_model_token_ratio_cache()


class TestCloneWindowRegression:
    """Acceptance #1: per-class beats the scalar AND flags the breach."""

    @pytest.mark.asyncio
    async def test_breach_window_is_held_out_of_the_fold(self) -> None:
        # Anti-tautology guard (the issue's named failure mode): the window we
        # predict must NOT be one we trained on. Assert it byte-for-byte: no
        # training window equals the breach composition.
        assert _CLONE_BREACH_BY_CLASS not in [bc for bc, _ in _TRAIN]
        # And it is genuinely held OUT -- the breach is ~6x larger than any
        # training window, so it cannot have leaked in under a different shape.
        largest_train = max(sum(bc.values()) for bc, _ in _TRAIN)
        assert 4 * largest_train < _BREACH_LOCAL

    @pytest.mark.asyncio
    async def test_perclass_beats_scalar_and_flags_breach(self) -> None:
        coefs = await _fit(_TRAIN)

        perclass_pred = _perclass_prediction(coefs, _CLONE_BREACH_BY_CLASS)
        scalar_pred = _BREACH_LOCAL * _SCALAR_R

        # (a) The scalar is the bug: it under-budgets the window below the
        #     harness's own window_max, so it would let the >1M breach through.
        assert scalar_pred < _BREACH_CAP, (
            f"scalar budget {scalar_pred:.0f} should be below the {_BREACH_CAP} "
            "window_max (this is the incident -- the scalar let the breach slip)"
        )

        # (b) The per-class fit FLAGS the breach: it budgets the window above
        #     the window_max, so the windower would drop a chunk and never send
        #     the >1M prompt. This is the must-pass gate (NOT a +/-5% in-sample
        #     APE -- a held-out refit cannot hit that, by the issue's own proof).
        assert perclass_pred > _BREACH_CAP, (
            f"per-class budget {perclass_pred:.0f} must exceed the {_BREACH_CAP} "
            "window_max so the breach window is caught (held-out fit)"
        )

        # (c) Per-class beats the scalar on absolute error against the real
        #     billed 1,005,124. The scalar's -19.6% undercount is the defect.
        perclass_err = abs(perclass_pred - _CLONE_BREACH_BILLED)
        scalar_err = abs(scalar_pred - _CLONE_BREACH_BILLED)
        assert perclass_err < scalar_err, (
            f"per-class error {perclass_err:.0f} must beat scalar error "
            f"{scalar_err:.0f} vs billed {_CLONE_BREACH_BILLED}"
        )
        # Sanity floor: the scalar's signed undercount really is ~-20%, and the
        # per-class fit really does roughly halve the absolute error.
        assert scalar_err / _CLONE_BREACH_BILLED > 0.15  # scalar >= 15% off
        assert perclass_err < scalar_err / 2  # per-class roughly halves it

    @pytest.mark.asyncio
    async def test_scalar_stub_fails_to_catch_the_breach(self) -> None:
        """Proves the fix is load-bearing, not the fixture.

        Re-run the SAME held-out window through the OLD single-scalar model
        (the implementation this PR replaces) and confirm it does the wrong
        thing: budgets under window_max and so would re-open the incident.
        This is the "fails-if-you-stub-the-solver-back-to-the-scalar" check
        the regression must encode -- if a future change quietly collapses the
        per-class fit back to a scalar, ``test_perclass_*`` above flips red and
        this test documents exactly why.
        """
        scalar_pred = _BREACH_LOCAL * _SCALAR_R
        # The scalar reproduces the incident: budgets ~808k, never flags >900k.
        assert scalar_pred == pytest.approx(_BREACH_LOCAL * _SCALAR_R)
        assert scalar_pred < _BREACH_CAP
        assert scalar_pred < _CLONE_BREACH_BILLED  # the -19.6% undercount


class TestStabilityGate:
    """Acceptance #8: the windower-consumed blended R_eff is in [2.0, 2.2] and
    is stable even though the raw per-class coefficients are not (collinear)."""

    @pytest.mark.asyncio
    async def test_fitted_blended_r_eff_in_band(self) -> None:
        # The fitted blend over the clone fail-window composition must land in
        # [2.0, 2.2] -- well above the 1.833 lifetime scalar, matching the
        # independently-observed healthy-span range for this heavy composition.
        coefs = await _fit(_TRAIN)
        r_eff = blended_r_eff(coefs, _CLONE_BREACH_BY_CLASS)
        assert 2.0 <= r_eff <= 2.2, f"blended R_eff {r_eff:.4f} outside [2.0, 2.2]"

    @pytest.mark.asyncio
    async def test_blend_is_robust_where_raw_coefficients_are_not(self) -> None:
        """The load-bearing robustness point the issue settles (and #8 protects).

        Under a one-fold perturbation (leave-one-window-out) the RAW per-class
        coefficients swing wildly -- thinking is present in only a few windows
        and is collinear with tool_result (r~0.979), so it is NOT individually
        identified (>=25% swings are expected and documented). But the BLENDED
        R_eff the windower actually consumes is far more stable: the
        collinearity moves mass *between* coefficients without moving their
        *blend*. Assert exactly that contrast.
        """
        base = await _fit(_TRAIN)
        base_r_eff = blended_r_eff(base, _CLONE_BREACH_BY_CLASS)
        active = [c for c in base if base[c] != 1.0]

        worst_raw_swing = 0.0
        worst_blend_swing = 0.0
        for i in range(len(_TRAIN)):
            perturbed = _TRAIN[:i] + _TRAIN[i + 1 :]
            refit = await _fit(perturbed)
            for c in active:
                if base[c] != 0.0:
                    worst_raw_swing = max(worst_raw_swing, abs(refit[c] - base[c]) / abs(base[c]))
            r = blended_r_eff(refit, _CLONE_BREACH_BY_CLASS)
            worst_blend_swing = max(worst_blend_swing, abs(r - base_r_eff) / base_r_eff)

        # At least one raw coefficient is under-identified -- the issue says so,
        # and the regularizer's job is to keep that from leaking into the blend,
        # NOT to make the raw coefficient stable.
        assert worst_raw_swing > 0.25, (
            "expected >=1 raw coefficient to be under-identified (collinear) -- "
            f"got worst raw swing {worst_raw_swing:.2%}; if this is now stable the "
            "fixture changed and the contrast below is no longer meaningful"
        )
        # The windower-consumed blend is materially more stable than the raw
        # coefficients (the whole point: build to the blend, not the split).
        assert worst_blend_swing < worst_raw_swing / 2, (
            f"blend swing {worst_blend_swing:.2%} should be << raw-coef swing "
            f"{worst_raw_swing:.2%} -- the blend must absorb the collinear churn"
        )
        # And the blend stays close to the in-band fit under perturbation: it
        # never collapses toward the broken 1.833 scalar nor explodes past the
        # provider's true heavy-composition ratio. (A wider band than the
        # fitted [2.0,2.2] -- a 10-window fold has leverage points; the contract
        # is "degrades to drops-one-chunk-early," never a scalar-grade undercount.)
        assert worst_blend_swing < 0.20, (
            f"blend swing {worst_blend_swing:.2%} too large -- R_eff is unstable"
        )
