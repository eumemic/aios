"""Trusted caller provenance propagated through request/job task contexts."""
from __future__ import annotations

import contextvars
from typing import Literal

from pydantic import BaseModel

ActorType = Literal["api_actor", "session_actor"]


class Actor(BaseModel):
    type: ActorType
    api_key_id: str | None = None
    session_id: str | None = None


_current: contextvars.ContextVar[Actor | None] = contextvars.ContextVar("current_actor", default=None)


def set_api_actor(key_id: str) -> None:
    _current.set(Actor(type="api_actor", api_key_id=key_id))


def set_session_actor(session_id: str) -> None:
    _current.set(Actor(type="session_actor", session_id=session_id))


def current_actor() -> Actor | None:
    return _current.get()


def actor_columns() -> tuple[str | None, str | None]:
    actor = current_actor()
    if actor is None:
        return None, None
    return actor.type, actor.session_id if actor.type == "session_actor" else actor.api_key_id


def actor_from_row(row: object) -> Actor | None:
    actor_type = row["created_by_type"]  # type: ignore[index]
    ref = row["created_by_ref"]  # type: ignore[index]
    if actor_type is None:
        return None
    if actor_type == "session_actor":
        return Actor(type=actor_type, session_id=ref)
    return Actor(type="api_actor", api_key_id=ref)
