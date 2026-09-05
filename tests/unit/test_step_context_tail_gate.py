"""The channels-listing gate (``tail_owes_response``).

The channels listing is a durable reminder row written when it changes (see
``aios.harness.reminders``), but never while the build ends on a *direct*
stimulus: a "0 unread" listing as the literal last line makes literal-minded
models (claude-fable-5) emit an empty turn instead of answering. The gate
keys on the structural tail class ``build_messages`` reports — never on the
rendered prose — and treats the trailing-stimulus notice as owed, whether it
is about to be written (``needs_trailing_notice``) or already a durable row
at the tail (``notice``): the missed events it points at ARE the stimulus.
The other reminder rows never reach this gate — ``build_messages`` classifies
the tail past them — so a row a failed attempt left behind an unanswered
inbound cannot flip it on the retry.
"""

from __future__ import annotations

from typing import get_args

import pytest

from aios.harness.context import TailOrigin
from aios.harness.reminders import tail_owes_response

_ALL_ORIGINS: tuple[TailOrigin, ...] = get_args(TailOrigin)


class TestTailOwesResponse:
    @pytest.mark.parametrize("origin", ["user", "tool", "notice"])
    def test_direct_stimulus_tail_owes(self, origin: TailOrigin) -> None:
        assert tail_owes_response(origin, needs_trailing_notice=False) is True

    @pytest.mark.parametrize("origin", ["assistant", "notification", "system", "none"])
    def test_non_stimulus_tail_does_not_owe(self, origin: TailOrigin) -> None:
        # A non-focal 🔔 marker is a navigation prompt whose companion IS the
        # listing; an assistant tail is the idle/sweep re-check where the
        # channel status is the useful signal; a system-only or empty build
        # carries no stimulus at all.
        assert tail_owes_response(origin, needs_trailing_notice=False) is False

    @pytest.mark.parametrize("origin", _ALL_ORIGINS)
    def test_trailing_notice_counts_as_owed_on_every_tail(self, origin: TailOrigin) -> None:
        assert tail_owes_response(origin, needs_trailing_notice=True) is True

    def test_every_tail_origin_is_classified(self) -> None:
        # A new TailOrigin literal must be placed on one side of the gate
        # deliberately; this pins that the gate is total over the type.
        assert set(_ALL_ORIGINS) == {
            "none",
            "system",
            "assistant",
            "tool",
            "user",
            "notification",
            "notice",
        }
