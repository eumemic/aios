"""Unit coverage for :class:`aios.sandbox.github_clone_breaker.GithubCloneBreaker`.

Pins the breaker contract issue #1720 specifies:

- Opens at K consecutive failures (not before).
- Exactly one DOWN-edge signal per degradation episode (dedup until recovery).
- ``is_open`` gates further attempts while the cooldown window is live.
- The cooldown lapsing allows exactly one half-open probe; success fully
  recovers, failure reopens with a fresh window and no duplicate DOWN edge.
- ``clear`` (the credential-rotation re-probe lever) resets state immediately
  regardless of any open window.
- Healthy repos (never failing) never enter the degraded set.
"""

from __future__ import annotations

import time

from aios.sandbox.github_clone_breaker import (
    _BREAKER_AUTH_BACKOFF_S,
    _BREAKER_FAILURE_THRESHOLD,
    _BREAKER_TRANSIENT_BACKOFF_S,
    GithubCloneBreaker,
)

_REPO = "ghrepo_01TEST"
_URL = "https://github.com/acme/foo.git"


def test_breaker_stays_closed_below_threshold() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD - 1):
        down = breaker.record_failure(_REPO, _URL)
        assert down is False
        assert breaker.is_open(_REPO) is False
    assert breaker.degraded_repos() == []


def test_breaker_opens_at_threshold_with_single_down_edge() -> None:
    breaker = GithubCloneBreaker()
    down_edges = [breaker.record_failure(_REPO, _URL) for _ in range(_BREAKER_FAILURE_THRESHOLD)]
    # Only the K-th failure is the DOWN transition.
    assert down_edges == [False] * (_BREAKER_FAILURE_THRESHOLD - 1) + [True]
    assert breaker.is_open(_REPO) is True
    degraded = breaker.degraded_repos()
    assert len(degraded) == 1
    assert degraded[0].resource_id == _REPO
    assert degraded[0].repo_url == _URL


def test_further_failures_while_open_do_not_re_signal_down() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL)
    assert breaker.is_open(_REPO) is True
    # One more failure while already open (shouldn't normally be recorded
    # since is_open gates the attempt, but the breaker itself must still
    # dedup if a caller calls it anyway).
    assert breaker.record_failure(_REPO, _URL) is False


def test_healthy_repo_never_degrades() -> None:
    breaker = GithubCloneBreaker()
    breaker.record_success(_REPO)
    assert breaker.is_open(_REPO) is False
    assert breaker.degraded_repos() == []


def test_record_success_closes_breaker() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL)
    assert breaker.is_open(_REPO) is True

    breaker.record_success(_REPO)

    assert breaker.is_open(_REPO) is False
    assert breaker.degraded_repos() == []


def test_cooldown_lapse_allows_half_open_probe() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL)
    opened_at = breaker._unhealthy_until[_REPO]

    # Before the cooldown elapses: still open, no probe allowed.
    assert breaker.is_open(_REPO, now=opened_at - 1.0) is True
    # Still shows as degraded for the prelude surface throughout the window.
    assert len(breaker.degraded_repos()) == 1

    # After the cooldown elapses: is_open returns False (probe allowed) but
    # the resource stays in the degraded set until the probe actually
    # succeeds.
    assert breaker.is_open(_REPO, now=opened_at + 1.0) is False
    assert len(breaker.degraded_repos()) == 1


def test_half_open_probe_failure_reopens_without_duplicate_down_edge() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL)
    opened_at = breaker._unhealthy_until[_REPO]
    assert breaker.is_open(_REPO, now=opened_at + 1.0) is False  # enters half-open

    # The probe clone fails again — reopens immediately (no need to
    # accumulate K failures again), and this is NOT a new DOWN edge (the
    # resource was already degraded).
    down_edge = breaker.record_failure(_REPO, _URL)
    assert down_edge is False
    assert breaker.is_open(_REPO) is True
    assert len(breaker.degraded_repos()) == 1


def test_half_open_probe_success_fully_recovers() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL)
    opened_at = breaker._unhealthy_until[_REPO]
    assert breaker.is_open(_REPO, now=opened_at + 1.0) is False  # enters half-open

    breaker.record_success(_REPO)

    assert breaker.is_open(_REPO) is False
    assert breaker.degraded_repos() == []
    # A subsequent failure series starts fresh from zero (not still half-open).
    for _ in range(_BREAKER_FAILURE_THRESHOLD - 1):
        assert breaker.record_failure(_REPO, _URL) is False
    assert breaker.is_open(_REPO) is False


def test_clear_resets_state_regardless_of_open_window() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL)
    assert breaker.is_open(_REPO) is True

    breaker.clear(_REPO)

    assert breaker.is_open(_REPO) is False
    assert breaker.degraded_repos() == []
    # A fresh failure series requires the full K again (not an instant
    # reopen the way a half-open probe failure is) since the credential
    # actually changed.
    for _ in range(_BREAKER_FAILURE_THRESHOLD - 1):
        assert breaker.record_failure(_REPO, _URL) is False
    assert breaker.is_open(_REPO) is False


def test_independent_resource_ids_have_independent_breaker_state() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure("ghrepo_dead", _URL)
    assert breaker.is_open("ghrepo_dead") is True
    assert breaker.is_open("ghrepo_healthy") is False
    assert len(breaker.degraded_repos()) == 1


def test_backoff_constants_positive_and_threshold_is_at_least_two() -> None:
    # Sanity guard against an accidental K=1 (breaker on first failure,
    # useless for transient blips) or a zero/negative cooldown.
    assert _BREAKER_FAILURE_THRESHOLD >= 2
    assert _BREAKER_AUTH_BACKOFF_S > 0
    assert _BREAKER_TRANSIENT_BACKOFF_S > 0
    # A transient blip must re-probe faster than a dead credential, else the
    # #1720 regression (1h lockout after a recovered outage) returns.
    assert _BREAKER_TRANSIENT_BACKOFF_S < _BREAKER_AUTH_BACKOFF_S


def test_auth_failure_opens_long_cooldown_and_records_cause() -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(_REPO, _URL, auth_failure=True, last_error="Authentication failed")
    # Auth failures earn the long (1h) cooldown.
    remaining = breaker._unhealthy_until[_REPO] - time.monotonic()
    assert remaining > _BREAKER_TRANSIENT_BACKOFF_S
    assert remaining <= _BREAKER_AUTH_BACKOFF_S + 1.0
    degraded = breaker.degraded_repos()[0]
    assert degraded.auth_failure is True
    assert degraded.last_error == "Authentication failed"


def test_transient_failure_opens_short_cooldown_not_the_hour_lockout() -> None:
    """The seat-gate regression fix: a transient (timeout/network/5xx) failure
    must NOT sit out the 1h lockout — it gets the short cooldown."""
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(
            _REPO, _URL, auth_failure=False, last_error="git clone timed out after 30.0s"
        )
    remaining = breaker._unhealthy_until[_REPO] - time.monotonic()
    assert remaining <= _BREAKER_TRANSIENT_BACKOFF_S + 1.0
    degraded = breaker.degraded_repos()[0]
    assert degraded.auth_failure is False
    assert degraded.last_error == "git clone timed out after 30.0s"


def test_half_open_reopen_refreshes_cause_and_backoff_class() -> None:
    """A half-open probe that fails for a NEW reason refreshes the recorded
    cause/backoff-class while keeping the original ``since``."""
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(
            _REPO, _URL, auth_failure=False, last_error="git clone timed out after 30.0s"
        )
    original_since = breaker.degraded_repos()[0].since
    opened_at = breaker._unhealthy_until[_REPO]
    assert breaker.is_open(_REPO, now=opened_at + 1.0) is False  # enters half-open

    # The probe now fails with an auth error — reopen with the long cooldown.
    assert (
        breaker.record_failure(_REPO, _URL, auth_failure=True, last_error="Authentication failed")
        is False
    )
    degraded = breaker.degraded_repos()[0]
    assert degraded.auth_failure is True
    assert degraded.last_error == "Authentication failed"
    assert degraded.since == original_since  # degradation start is preserved
    remaining = breaker._unhealthy_until[_REPO] - time.monotonic()
    assert remaining > _BREAKER_TRANSIENT_BACKOFF_S
