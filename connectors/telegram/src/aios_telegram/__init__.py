"""Telegram connector for aios."""

from __future__ import annotations

from .config import Settings
from .connector import TelegramConnector


def make_connector() -> TelegramConnector:
    """Entry point resolved by the aios connector supervisor."""
    return TelegramConnector(Settings())
