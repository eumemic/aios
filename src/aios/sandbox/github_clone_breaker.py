"""Per-repo circuit breaker for host-side github clone attempts (#1720).

Production evidence: a session with a dead/expired token on an attached
``github_repository`` retried the clone on EVERY turn, forever — 2,100
``github_clone_failed`` events in 24h on one session, each turn paying the
failed-clone latency in its prelude, plus needless auth-failure traffic
against GitHub from our egress IP.

This module is the worker-process, in-memory breaker that stops that churn:
after ``_BREAKER_FAILURE_THRESHOLD`` (K) consecutive clone failures for one
attachment (keyed on the ``ghrepo_...`` resource id — unique per session
attachment, so breaker state never crosses sessions, and stable across a
token rotation on the SAME attachment — see :func:`aios.services.
github_repositories.rotate_token`), the breaker opens for a cooldown whose
length depends on the failure CLASS — ``_BREAKER_AUTH_BACKOFF_S`` (1h) for an
auth-shaped failure (dead/expired token, which won't fix itself) vs
``_BREAKER_TRANSIENT_BACKOFF_S`` (60s) for a timeout/network/GitHub-5xx blip
that should re-probe fast — and :func:`aios.sandbox.spec.
_materialize_github_clones` skips the clone attempt entirely rather than
re-hitting a dead credential every step.

Deliberately mirrors :mod:`aios.mcp.pool`'s per-key breaker (#1698) — same
threshold, same cooldown/half-open re-probe shape — so the two breakers
read as one shared pattern across the codebase. This one is simpler: the
caller (``_materialize_github_clones``) already runs WITH the session/
account context the failure needs to log, so there's no pool-level
"record now, drain-and-emit later from a session-aware caller" indirection
to bridge — :meth:`record_failure`'s return value tells the caller directly
whether this is the DOWN edge to emit ``github_clone_degraded`` for.

Half-open re-probe (issue #1720's proposed fix, path (c) "slow timer"):
when the cooldown window lapses, :meth:`is_open` allows exactly ONE probe
attempt through ("half-open"). If that probe fails, the breaker reopens
immediately with a fresh cooldown (no need to accumulate K failures again,
and no duplicate DOWN event — the resource was already marked degraded).
If it succeeds, :meth:`record_success` fully closes the breaker.

The other two re-probe paths from the issue:

(a) **Credential row update** — :func:`aios.services.github_repositories.
    rotate_token` calls :meth:`GithubCloneBreaker.clear` after a successful
    UPDATE, so a fixed token re-probes on the very next provision instead of
    serving out a cooldown opened under the old secret.
(b) **repo_url update** — url/mount_path are immutable on an existing
    attachment (the API only rotates the token/identity); changing the repo
    means detach + a fresh attach, which mints a brand-new resource id and
    therefore a fresh (closed) breaker key automatically.
(d) **Explicit agent/API request** — :meth:`clear` is also the primitive an
    explicit re-probe request would call (e.g. a future "retry now" tool/
    endpoint); not wired to a caller yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from aios.logging import get_logger

log = get_logger("aios.sandbox.github_clone_breaker")

# Consecutive clone failures for one resource id before the breaker opens.
# Mirrors ``aios.mcp.pool._BREAKER_FAILURE_THRESHOLD`` (#1698) — same K
# across both breaker implementations by design.
_BREAKER_FAILURE_THRESHOLD = 3

# Cooldown for an AUTH-SHAPED failure (dead/expired/invalid token). An hour,
# per issue #1720's proposed "(c) slow timer (e.g. hourly)" re-probe — long
# enough that a dead credential doesn't hammer GitHub's auth endpoint, short
# enough that a same-day credential fix is picked up without an explicit
# re-probe.
_BREAKER_AUTH_BACKOFF_S = 3600.0

# Cooldown for a TRANSIENT failure (timeout / network / GitHub 5xx). Short —
# mirrors the MCP breaker this one is modeled on
# (``aios.mcp.pool._BREAKER_UNHEALTHY_BACKOFF_S`` = 60s) — so a GitHub blip or
# a provision-time network hiccup re-probes within a minute instead of
# sitting out a full hour AFTER the outage clears. The 1h lockout on a
# recovered-transient failure was the regression the #1720 seat gate caught:
# pre-PR the next step retried instantly.
_BREAKER_TRANSIENT_BACKOFF_S = 60.0


@dataclass(slots=True)
class DegradedRepo:
    """One breaker-open repo attachment, for the prelude/agent-visible surface.

    ``since`` is a wall-clock UTC timestamp (not ``time.monotonic()``) —
    it's rendered directly into the prelude health line
    (``repos: <mount_path> AUTH-FAILED since <since>``), so it needs to
    mean something to a human/agent reading it, not just order events
    within one process's uptime.

    ``auth_failure`` + ``last_error`` (issue #1720 seat-gate fix) carry the
    real cause so the prelude renders the TRUTH — ``AUTH-FAILED`` only for a
    classified auth failure, else ``CLONE-FAILING ... (last: <err>)`` with the
    (already token-redacted) git error — instead of hardcoding ``AUTH-FAILED``
    for every degraded repo.
    """

    resource_id: str
    repo_url: str
    mount_path: str
    since: datetime
    auth_failure: bool = False
    last_error: str = ""


@dataclass(slots=True)
class GithubCloneBreaker:
    """Worker-process, in-memory circuit breaker keyed on github repo resource id.

    Not persisted across worker restarts — a fresh worker starts every
    breaker closed, which is the correct fail-open default (a dead
    credential is re-discovered within K failures, not silently muted
    forever by a stale in-memory flag from a previous process).
    """

    _failure_count: dict[str, int] = field(default_factory=dict)
    _unhealthy_until: dict[str, float] = field(default_factory=dict)
    _half_open: set[str] = field(default_factory=set)
    _degraded: dict[str, DegradedRepo] = field(default_factory=dict)

    def is_open(self, resource_id: str, *, now: float | None = None) -> bool:
        """True if this resource id is in a clone backoff window.

        Callers consult this BEFORE attempting a clone so a dead credential
        is skipped fast instead of re-stalling every step. On expiry the
        window clears and the resource enters "half-open": this call
        returns ``False`` (the caller should attempt exactly one probe
        clone) but the resource stays in :meth:`degraded_repos` until that
        probe actually succeeds — a cooldown lapse alone doesn't heal the
        prelude surface.
        """
        effective_now = now if now is not None else time.monotonic()
        until = self._unhealthy_until.get(resource_id)
        if until is None:
            return False
        if effective_now >= until:
            del self._unhealthy_until[resource_id]
            self._half_open.add(resource_id)
            return False
        return True

    def record_failure(
        self,
        resource_id: str,
        repo_url: str,
        mount_path: str = "",
        *,
        auth_failure: bool = False,
        last_error: str = "",
    ) -> bool:
        """Count one consecutive clone failure; open the breaker at K.

        Returns ``True`` on the DOWN transition (the failure that first
        opens the circuit) so the caller — which already has session
        context in hand — can emit exactly one ``github_clone_degraded``
        lifecycle event. A half-open probe failure reopens the breaker
        immediately (fresh cooldown) but returns ``False``: the resource
        was already degraded, so no duplicate event.

        ``auth_failure`` classifies THIS failure (issue #1720 seat-gate fix):
        an auth-shaped failure (dead/expired token) opens the long
        ``_BREAKER_AUTH_BACKOFF_S`` cooldown; a transient one (timeout /
        network / 5xx) the short ``_BREAKER_TRANSIENT_BACKOFF_S`` so it
        re-probes fast instead of eating a full-hour lockout after the outage
        clears. ``last_error`` is the (already token-redacted) git message
        carried into :class:`DegradedRepo` so the prelude renders the real
        cause rather than a hardcoded ``AUTH-FAILED``.
        """
        backoff_s = _BREAKER_AUTH_BACKOFF_S if auth_failure else _BREAKER_TRANSIENT_BACKOFF_S
        if resource_id in self._half_open:
            self._half_open.discard(resource_id)
            self._unhealthy_until[resource_id] = time.monotonic() + backoff_s
            # Refresh the degraded record's cause with this probe's failure so
            # the prelude line reflects the latest reason (keep original since).
            existing = self._degraded.get(resource_id)
            if existing is not None:
                existing.auth_failure = auth_failure
                existing.last_error = last_error
            log.warning(
                "github_clone_breaker.reopened",
                resource_id=resource_id,
                repo_url=repo_url,
                backoff_s=backoff_s,
                auth_failure=auth_failure,
            )
            return False

        count = self._failure_count.get(resource_id, 0) + 1
        self._failure_count[resource_id] = count
        if count < _BREAKER_FAILURE_THRESHOLD:
            return False

        self._unhealthy_until[resource_id] = time.monotonic() + backoff_s
        transitioned_down = resource_id not in self._degraded
        if transitioned_down:
            self._degraded[resource_id] = DegradedRepo(
                resource_id=resource_id,
                repo_url=repo_url,
                mount_path=mount_path or repo_url,
                since=datetime.now(UTC),
                auth_failure=auth_failure,
                last_error=last_error,
            )
            log.warning(
                "github_clone_breaker.opened",
                resource_id=resource_id,
                repo_url=repo_url,
                failures=count,
                backoff_s=backoff_s,
                auth_failure=auth_failure,
            )
        return transitioned_down

    def record_success(self, resource_id: str) -> None:
        """Clear breaker state for a successful clone — re-closes the circuit."""
        self._failure_count.pop(resource_id, None)
        self._unhealthy_until.pop(resource_id, None)
        self._half_open.discard(resource_id)
        self._degraded.pop(resource_id, None)

    def clear(self, resource_id: str) -> None:
        """Explicitly reset all breaker state for ``resource_id``.

        Called on a credential row update (rotate_token) and available for
        an explicit agent/API re-probe request (issue #1720 (a)/(d)) — the
        next clone attempt re-probes immediately regardless of any open
        cooldown window, and (unlike the half-open path) is treated as a
        brand-new failure series if it fails again (full K threshold, not
        an instant reopen) since the credential actually changed.
        """
        self._failure_count.pop(resource_id, None)
        self._unhealthy_until.pop(resource_id, None)
        self._half_open.discard(resource_id)
        self._degraded.pop(resource_id, None)

    def degraded_repos(self) -> list[DegradedRepo]:
        """Return every resource id whose breaker is currently OPEN or half-open.

        In-memory, per-worker accessor for tests/introspection and the
        prelude health surface — reflects only this worker's breaker state.
        """
        return list(self._degraded.values())


__all__ = [
    "_BREAKER_AUTH_BACKOFF_S",
    "_BREAKER_FAILURE_THRESHOLD",
    "_BREAKER_TRANSIENT_BACKOFF_S",
    "DegradedRepo",
    "GithubCloneBreaker",
]
