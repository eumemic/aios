"""Exhaustiveness/coverage pin for ``_inbound_drop_error`` (#1544).

The status mapping is now a total ``match`` over :class:`InboundDrop` ending in
``assert_never``; ``set(_EXPECTED) == set(InboundDrop)`` forces any future enum
member to be wired through both this table and the ``match`` (or mypy fails first).
"""

import pytest

from aios.api.routers.connectors import _inbound_drop_error
from aios.services.inbound import InboundDrop

_EXPECTED = {
    InboundDrop.PAYLOAD_TOO_LARGE: 413,
    InboundDrop.DETACHED: 422,
    InboundDrop.ARCHIVED_TEMPLATE: 422,
    InboundDrop.DENIED_BY_POLICY: 422,
    InboundDrop.RATE_LIMITED: 429,
    InboundDrop.ATTACHMENT_STAGING_FAILED: 500,
    InboundDrop.SESSION_MISSING: 404,
}


def test_every_inbound_drop_member_has_an_explicit_status() -> None:
    # Guards the whole enum: a new arm with no mapping fails here AND at mypy.
    assert set(_EXPECTED) == set(InboundDrop)


@pytest.mark.parametrize("reason, status", _EXPECTED.items())
def test_inbound_drop_status_mapping(reason: InboundDrop, status: int) -> None:
    err = _inbound_drop_error(reason)
    assert err.status_code == status
    assert err.detail == {"drop_reason": reason.value}
