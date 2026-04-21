"""Exception hierarchy for aios-telegram."""

from __future__ import annotations


class TelegramConnectorError(Exception):
    """Base class for all aios-telegram errors."""


class BotIdentityError(TelegramConnectorError):
    """Raised when we can't resolve the bot's numeric id at startup."""
