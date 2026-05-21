"""Time-format helpers shared by connector implementations."""

from __future__ import annotations

from datetime import UTC, datetime


def iso_from_ms(timestamp_ms: int) -> str:
    """Return the ISO-8601 UTC string for ``timestamp_ms`` milliseconds.

    The connector wire boundary (:meth:`HttpConnector.emit_inbound`'s
    ``timestamp`` argument) accepts an ISO-8601 string; platform SDKs
    typically expose millisecond integers.  Centralising the conversion
    keeps the precision (microsecond) and timezone (UTC) consistent
    across connectors.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
