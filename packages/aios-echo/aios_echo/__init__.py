"""Echo connector — canonical example of an aios connector.

Two tools (``ping``, ``echo``) and one tool (``trigger_inbound``) that
synthesizes an inbound message so e2e tests can exercise the
spool/dedup/append flow without spinning up a real platform daemon.

The connector ships as a separate published package so it can serve
as both an installable example for connector authors and the test
fixture aios runs in CI to verify the end-to-end pipeline.
"""

from __future__ import annotations

from aios_echo.connector import EchoConnector


def make_connector() -> EchoConnector:
    """Entry point resolved by the aios connector supervisor."""
    return EchoConnector()


__all__ = ["EchoConnector", "make_connector"]
