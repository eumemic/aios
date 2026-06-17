"""Slack connector for aios (#1229).

MVP slices 1-3/4 — the connection layer (package scaffold, Socket-Mode
transport, ``serve_connection`` lifecycle), the inbound decision layer
(normalization + the four connector gates), and the outbound reply layer
(the ``slack_send`` / ``slack_react`` ``@tool``\\ s + the markdown→mrkdwn
pipeline and hard clamps).  A live DM-round-trip smoke lands in slice D.
"""

from __future__ import annotations

from .connector import SlackConnector

__all__ = ["SlackConnector"]
