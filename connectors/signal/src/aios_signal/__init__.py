"""aios-signal: Signal connector for aios."""

from __future__ import annotations

from .config import Settings
from .connector import SignalConnector


def make_connector() -> SignalConnector:
    """Entry point resolved by the aios connector supervisor."""
    return SignalConnector(Settings())
