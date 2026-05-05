"""Connector launch description.

A :class:`ConnectorSpec` tells the supervisor how to spawn one
connector subprocess: which executable, which args, which env, which
cwd.  The supervisor builds the spec internally from the
``aios.connectors`` entry-point name — connector authors don't
construct :class:`ConnectorSpec` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConnectorSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: Path | None = None
