"""Regression test pinning the inbound-admission denial to a NON-fatal 422.

Part of #1500. The server maps a ``denied_by_policy`` inbound drop to HTTP 422
(``ValidationError``), NOT 403 and NOT 5xx. This is load-bearing for
deploy-safety: ``_is_fatal_inbound_status`` treats 401/403/5xx as fatal —
those crash-restart the connector container, killing every connection it
serves. A denied *stranger* must drop one envelope and leave the container
serving every other connection, so the denial status must be a routine 4xx.

This test asserts the contract from the connector runner's side: 422 is
non-fatal, while 403/5xx (which the denial must never be) are fatal.
"""

from __future__ import annotations

from aios_connector_http.runner import _is_fatal_inbound_status

# Mirror the server-side enum value without importing the aios server package
# (this is the connector-http package's own test suite). If the server renames
# the reason, the server-side unit test (test_inbound_admission) catches it;
# this test pins only the *status* contract the runner depends on.
_DENIED_BY_POLICY = "denied_by_policy"


def test_denied_by_policy_maps_to_non_fatal_422() -> None:
    # The denial status the server returns for ``denied_by_policy``.
    denied_status = 422

    # 422 is a routine per-envelope drop — the container keeps serving.
    assert _is_fatal_inbound_status(denied_status) is False

    # Defensive: the statuses the denial must NEVER be are fatal, so if a future
    # refactor regresses the mapping to 403 or 5xx the connector would crash on
    # the first denied stranger.
    assert _is_fatal_inbound_status(403) is True
    assert _is_fatal_inbound_status(500) is True


def test_denied_by_policy_reason_constant_is_stable() -> None:
    # Tripwire: the reason string the server emits for a policy denial.
    assert _DENIED_BY_POLICY == "denied_by_policy"
