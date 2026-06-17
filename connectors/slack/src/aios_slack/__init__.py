"""Slack connector for aios (#1229).

MVP slice 1/4 — the connection layer: package scaffold, Socket-Mode
transport, and the ``serve_connection`` lifecycle.  Parsing, gating,
and the outbound ``@tool`` vocabulary land in later slices.
"""

from __future__ import annotations

from .connector import SlackConnector

__all__ = ["SlackConnector"]
