"""idempotency_key: the single per-call token derivation, shared by both deliveries.

The byte-identity tests here PIN the wire contract: the sandbox/``bash`` env preamble
and the worker/``http_request`` sentinel substitution both derive from this one
function, so a drift in either path is a silent dedup failure. Cross-path equality is
the real guard.
"""

from __future__ import annotations

import hashlib

from aios.workflows.idempotency_key import (
    AIOS_IDEMPOTENCY_KEY_SENTINEL,
    idempotency_key,
)


def test_derivation_is_sha256_of_nul_separated_pair() -> None:
    # Byte-identity: the token IS sha256(run_id\0call_key) hexdigest. This pins the
    # wire format — any change here breaks dedup for every effect already in flight.
    run_id = "wfr_1"
    call_key = "sha:deadbeef#2"
    expected = hashlib.sha256(b"wfr_1\0sha:deadbeef#2").hexdigest()
    assert idempotency_key(run_id, call_key) == expected


def test_same_call_is_stable_across_calls() -> None:
    # A crash re-drive carries the SAME run_id + call_key → byte-identical token.
    assert idempotency_key("wfr_9", "sha:abc#0") == idempotency_key("wfr_9", "sha:abc#0")


def test_distinct_call_keys_give_distinct_tokens() -> None:
    # Two distinct mutating calls in one run must NOT share a token (else the upstream
    # silently dedups the second — the per-run bug the CEO decision rules out).
    assert idempotency_key("wfr_9", "sha:abc#0") != idempotency_key("wfr_9", "sha:abc#1")


def test_distinct_run_ids_give_distinct_tokens() -> None:
    # A new/cloned run re-fires the effect deliberately → distinct token.
    assert idempotency_key("wfr_1", "sha:abc#0") != idempotency_key("wfr_2", "sha:abc#0")


def test_nul_separator_prevents_boundary_collision() -> None:
    # The NUL separator makes (run_id, call_key) unambiguous: a ":" or "" join would
    # let ("a", "bc") and ("ab", "c") collide. NUL never appears in a ULID run_id.
    assert idempotency_key("ab", "c") != idempotency_key("a", "bc")


def test_sentinel_mirrors_the_bash_env_var() -> None:
    # The worker http_request opt-in sentinel mirrors the bash path's env var name,
    # so authors learn one ergonomic across both deliveries.
    assert AIOS_IDEMPOTENCY_KEY_SENTINEL == "$AIOS_IDEMPOTENCY_KEY"
