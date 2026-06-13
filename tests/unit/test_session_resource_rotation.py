"""Unit coverage for per-resource github_repository token rotation (#1029).

Rotating ONE attached repo's token must not require re-supplying every
other repo's secret (the full-list-replace ``PUT`` path). The per-resource
route ``POST /v1/sessions/{sid}/resources/{rid}`` does an in-place UPDATE on
the ``session_github_repositories`` row, scoped to ``(session_id, id,
account_id)``:

* the attachment id is the unified ``ghrepo_`` namespace — there is no
  separate ``sghr_`` namespace (one right way, no aliases);
* omitted ``git_user_name`` / ``git_user_email`` preserve the stored
  identity (partial update by design);
* the row's ``updated_at`` bumps, which is part of the mount snapshot key
  so the rotation reaches the sandbox on the next provision.

These behaviours are also covered end-to-end against real Postgres in
``tests/e2e/test_github_repositories.py``; this module pins them at the
unit boundary (route id-guard + service arg construction) without Docker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.api.routers.sessions import _require_github_resource_id, update_resource
from aios.errors import NotFoundError, ValidationError
from aios.ids import GITHUB_REPOSITORY, MEMORY_STORE, make_id
from aios.models.github_repositories import (
    GithubRepositoryResourceEcho,
    GithubRepositoryUpdate,
)
from aios.services import github_repositories as github_repo_service
from tests.unit.conftest import fake_pool_yielding_conn


def _echo(
    rid: str, *, updated_at: datetime, name: str | None, email: str | None
) -> GithubRepositoryResourceEcho:
    return GithubRepositoryResourceEcho(
        id=rid,
        url="https://github.com/octocat/Hello-World.git",
        mount_path="/workspace/repo",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=updated_at,
        git_user_name=name,
        git_user_email=email,
    )


def _fake_conn() -> Any:
    """A conn whose ``conn.transaction()`` works as an async ctx manager."""
    conn = MagicMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction.return_value = tx
    return conn


def _fake_crypto_box() -> Any:
    box = MagicMock()
    subkey = MagicMock()
    subkey.encrypt.return_value = MagicMock(name="EncryptedBlob")
    box.derive_account_subkey.return_value = subkey
    return box


# ─── route id-guard: ghrepo_ accepted, everything else 4xx ──────────────────


class TestRequireGithubResourceId:
    def test_accepts_session_attachment_ghrepo_id(self) -> None:
        """Session attachments live in the unified ``ghrepo_`` namespace,
        so the per-resource rotation route accepts them — no separate
        ``sghr_`` namespace (#1029)."""
        rid = make_id(GITHUB_REPOSITORY)
        assert rid.startswith("ghrepo_")
        # Does not raise.
        _require_github_resource_id(rid)

    def test_rejects_memory_store_id_with_validation_error(self) -> None:
        rid = make_id(MEMORY_STORE)
        with pytest.raises(ValidationError) as exc:
            _require_github_resource_id(rid)
        # Message steers the caller to the right prefix.
        assert "ghrepo_" in str(exc.value)

    def test_rejects_malformed_id_with_validation_error(self) -> None:
        with pytest.raises(ValidationError) as exc:
            _require_github_resource_id("not-a-valid-id")
        assert "malformed" in str(exc.value).lower()


# ─── route: partial-update identity construction ────────────────────────────


class TestUpdateResourceRoute:
    async def test_token_only_passes_identity_none(self) -> None:
        """Token-only rotation must preserve the stored identity: the route
        passes ``identity=None`` so the service leaves ``git_user_*`` alone."""
        rid = make_id(GITHUB_REPOSITORY)
        body = GithubRepositoryUpdate.model_validate({"authorization_token": "ghp_new"})
        captured: dict[str, Any] = {}

        async def _spy(*_a: Any, **kw: Any) -> GithubRepositoryResourceEcho:
            captured.update(kw)
            return _echo(rid, updated_at=datetime(2026, 6, 1, tzinfo=UTC), name="Old", email="o@x")

        with patch.object(github_repo_service, "rotate_token", AsyncMock(side_effect=_spy)):
            await update_resource(
                session_id="sess_x",
                resource_id=rid,
                body=body,
                pool=MagicMock(),
                crypto_box=MagicMock(),
                account_id="acc_1",
            )

        assert captured["identity"] is None
        assert captured["resource_id"] == rid
        assert captured["new_token"] == "ghp_new"
        assert captured["session_id"] == "sess_x"
        assert captured["account_id"] == "acc_1"

    async def test_identity_supplied_passes_tuple(self) -> None:
        rid = make_id(GITHUB_REPOSITORY)
        body = GithubRepositoryUpdate.model_validate(
            {
                "authorization_token": "ghp_new",
                "git_user_name": "New Name",
                "git_user_email": "new@x",
            }
        )
        captured: dict[str, Any] = {}

        async def _spy(*_a: Any, **kw: Any) -> GithubRepositoryResourceEcho:
            captured.update(kw)
            return _echo(
                rid, updated_at=datetime(2026, 6, 1, tzinfo=UTC), name="New Name", email="new@x"
            )

        with patch.object(github_repo_service, "rotate_token", AsyncMock(side_effect=_spy)):
            await update_resource(
                session_id="sess_x",
                resource_id=rid,
                body=body,
                pool=MagicMock(),
                crypto_box=MagicMock(),
                account_id="acc_1",
            )

        assert captured["identity"] == ("New Name", "new@x")

    async def test_partial_identity_passes_tuple_with_none(self) -> None:
        """Supplying only ``git_user_name`` still sends a tuple — the absent
        email is an explicit ``None`` (replace both atomically), not a
        preserve. Preserve only happens when neither identity field is set."""
        rid = make_id(GITHUB_REPOSITORY)
        body = GithubRepositoryUpdate.model_validate(
            {"authorization_token": "ghp_new", "git_user_name": "Only Name"}
        )
        captured: dict[str, Any] = {}

        async def _spy(*_a: Any, **kw: Any) -> GithubRepositoryResourceEcho:
            captured.update(kw)
            return _echo(
                rid, updated_at=datetime(2026, 6, 1, tzinfo=UTC), name="Only Name", email=None
            )

        with patch.object(github_repo_service, "rotate_token", AsyncMock(side_effect=_spy)):
            await update_resource(
                session_id="sess_x",
                resource_id=rid,
                body=body,
                pool=MagicMock(),
                crypto_box=MagicMock(),
                account_id="acc_1",
            )

        assert captured["identity"] == ("Only Name", None)

    async def test_rejects_non_github_id_before_calling_service(self) -> None:
        body = GithubRepositoryUpdate.model_validate({"authorization_token": "ghp_new"})
        spy = AsyncMock()
        with (
            patch.object(github_repo_service, "rotate_token", spy),
            pytest.raises(ValidationError),
        ):
            await update_resource(
                session_id="sess_x",
                resource_id=make_id(MEMORY_STORE),
                body=body,
                pool=MagicMock(),
                crypto_box=MagicMock(),
                account_id="acc_1",
            )
        spy.assert_not_awaited()


# ─── service: scoped UPDATE, 404 on miss, updated_at bump ───────────────────


class TestRotateTokenService:
    async def test_passes_scope_and_returns_bumped_echo(self) -> None:
        """The service re-encrypts under the account subkey and forwards the
        (session_id, resource_id, account_id) scope to the query; the echo it
        returns carries the bumped ``updated_at``."""
        rid = make_id(GITHUB_REPOSITORY)
        new_updated = datetime(2026, 6, 12, tzinfo=UTC)
        captured: dict[str, Any] = {}

        async def _update(
            _conn: Any, session_id: str, resource_id: str, blob: Any, **kw: Any
        ) -> GithubRepositoryResourceEcho:
            captured["session_id"] = session_id
            captured["resource_id"] = resource_id
            captured["blob"] = blob
            captured.update(kw)
            return _echo(rid, updated_at=new_updated, name="Old", email="o@x")

        conn = _fake_conn()
        pool = fake_pool_yielding_conn(conn)
        box = _fake_crypto_box()

        with patch(
            "aios.services.github_repositories.queries.update_session_github_repo_blob",
            AsyncMock(side_effect=_update),
        ):
            echo = await github_repo_service.rotate_token(
                pool,
                box,
                account_id="acc_1",
                session_id="sess_x",
                resource_id=rid,
                new_token="ghp_rotated",
                identity=None,
            )

        # Re-encryption is keyed to the account subkey, not the master key.
        box.derive_account_subkey.assert_called_once_with("acc_1")
        # Scope forwarded verbatim — foreign session/account simply won't match.
        assert captured["session_id"] == "sess_x"
        assert captured["resource_id"] == rid
        assert captured["account_id"] == "acc_1"
        assert captured["identity"] is None
        # The bumped updated_at flows back out (mount-snapshot key → recycle).
        assert echo.updated_at == new_updated
        assert echo.id == rid

    async def test_foreign_scope_propagates_not_found(self) -> None:
        """A foreign-account or foreign-session id matches no row, so the
        query raises ``NotFoundError`` (HTTP 404) — the service does not
        swallow it."""
        rid = make_id(GITHUB_REPOSITORY)
        conn = _fake_conn()
        pool = fake_pool_yielding_conn(conn)
        box = _fake_crypto_box()

        with (
            patch(
                "aios.services.github_repositories.queries.update_session_github_repo_blob",
                AsyncMock(
                    side_effect=NotFoundError(
                        f"github_repository resource {rid} not found on session sess_x",
                        detail={"session_id": "sess_x", "resource_id": rid},
                    )
                ),
            ),
            pytest.raises(NotFoundError),
        ):
            await github_repo_service.rotate_token(
                pool,
                box,
                account_id="acc_OTHER",
                session_id="sess_x",
                resource_id=rid,
                new_token="ghp_rotated",
                identity=None,
            )
