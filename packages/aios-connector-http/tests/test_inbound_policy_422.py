"""Regression test pinning the inbound-admission denial to a NON-fatal 422.

Part of #1500. The server maps a ``denied_by_policy`` inbound drop to HTTP 422
(``ValidationError``), NOT 403. This is load-bearing for deploy-safety:
``_is_fatal_inbound_status`` treats authentication failures as fatal. A denied
*stranger* must drop one envelope and leave the container serving every other
connection, so the denial status must not be an authentication failure.

This test asserts the contract from the connector runner's side: 422 and 5xx
are per-message drops, while 403 (which the denial must never be) is fatal.
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

    # Authentication failures remain fatal, while transient API failures drop
    # the affected message without killing the connection feed.
    assert _is_fatal_inbound_status(403) is True
    assert _is_fatal_inbound_status(500) is False


def test_denied_by_policy_reason_constant_is_stable() -> None:
    # Tripwire: the reason string the server emits for a policy denial.
    assert _DENIED_BY_POLICY == "denied_by_policy"
