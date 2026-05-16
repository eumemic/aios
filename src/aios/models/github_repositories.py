"""GitHub repository session resources.

Each ``github_repository`` is a per-session attachment that mounts a git repo
into the sandbox at a user-specified ``mount_path``. Unlike memory stores,
there is no top-level repository resource — the row in
``session_github_repositories`` IS the resource. This mirrors Anthropic
Managed Agents' wire shape exactly.

The clone token (``authorization_token``) is write-only on every endpoint:
required at create, settable via ``POST /v1/sessions/{sid}/resources/{rid}``
for rotation, never echoed back in API responses. Stored encrypted at rest
via the same libsodium CryptoBox used by ``vault_credentials``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

# Cap mirrors memory_stores' MAX_STORES_PER_SESSION. There's no hard
# resource constraint forcing a low cap; this is a sanity bound that
# matches the existing aios convention for per-session mountable resources.
MAX_REPOS_PER_SESSION = 8

# Mount path pattern: absolute path, segments may not be empty/contain NUL.
# Same shape as ``_MEMORY_PATH_PATTERN`` in models/memory_stores.py.
_MOUNT_PATH_PATTERN = r"^(/[^/\x00]+)+$"


def _check_mount_path(path: str) -> None:
    """Block path-traversal segments and reserved-mount overlaps."""
    for segment in path.lstrip("/").split("/"):
        if segment in (".", ".."):
            raise ValueError(f"path segment {segment!r} is not allowed (would enable traversal)")
    if path == "/mnt/memory" or path.startswith("/mnt/memory/"):
        raise ValueError("mount_path may not overlap with the /mnt/memory tree (memory stores)")
    if path == "/workspace":
        raise ValueError(
            "mount_path may not be /workspace exactly (it shadows the session workspace mount)"
        )


class GithubRepositoryResource(BaseModel):
    """Item in ``Session.resources[]`` request shape (create + update).

    The ``authorization_token`` is a write-only ``SecretStr`` — present on
    request, encrypted on the way to the DB, and never returned in API
    responses (see :class:`GithubRepositoryResourceEcho`).

    ``git_user_name`` / ``git_user_email`` are optional and stamped via
    ``git config`` after clone so commits made inside the sandbox carry
    a deterministic identity per resource without the agent needing to
    self-correct from git's "Please tell me who you are" error.  Absent
    means "no identity configured" — the v1 default.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["github_repository"]
    url: str = Field(min_length=1)
    mount_path: str = Field(min_length=2, max_length=4096, pattern=_MOUNT_PATH_PATTERN)
    authorization_token: SecretStr
    git_user_name: str | None = Field(default=None, max_length=256)
    git_user_email: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _check(self) -> GithubRepositoryResource:
        _check_mount_path(self.mount_path)
        if not self.authorization_token.get_secret_value():
            raise ValueError("authorization_token must be non-empty")
        return self


class GithubRepositoryResourceEcho(BaseModel):
    """Read view of an attached github repository as echoed on
    ``Session.resources`` and on the per-resource sub-collection endpoints.

    Token is intentionally absent; ``id`` is the per-attachment ULID
    (prefix ``ghrepo``) used by the rotation endpoint.  ``git_user_name``
    / ``git_user_email`` echo back as plaintext (they aren't secrets).
    """

    type: Literal["github_repository"] = "github_repository"
    id: str
    url: str
    mount_path: str
    created_at: datetime
    updated_at: datetime
    git_user_name: str | None = None
    git_user_email: str | None = None


class GithubRepositoryUpdate(BaseModel):
    """Request body for ``POST /v1/sessions/{sid}/resources/{rid}``.

    Token rotation is the primary action; ``git_user_name`` /
    ``git_user_email`` may also be supplied alongside, in which case the
    stored identity is replaced.  ``url`` and ``mount_path`` are
    immutable after creation — to change them, detach the resource and
    attach a new one.
    """

    model_config = ConfigDict(extra="forbid")

    authorization_token: SecretStr
    git_user_name: str | None = Field(default=None, max_length=256)
    git_user_email: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _check(self) -> GithubRepositoryUpdate:
        if not self.authorization_token.get_secret_value():
            raise ValueError("authorization_token must be non-empty")
        return self


def validate_resources(resources: list[GithubRepositoryResource]) -> None:
    """Cross-item invariants checked at the API boundary.

    The DB has a unique index on ``(session_id, mount_path)`` so a duplicate
    here would 500 with a constraint violation; catching it pre-DB lets us
    return a clean 4xx with a useful message.
    """
    if len(resources) > MAX_REPOS_PER_SESSION:
        raise ValueError(f"at most {MAX_REPOS_PER_SESSION} github repositories per session")
    seen_paths: set[str] = set()
    for resource in resources:
        if resource.mount_path in seen_paths:
            raise ValueError(f"duplicate mount_path {resource.mount_path!r}")
        seen_paths.add(resource.mount_path)
