"""Unit tests for github_repository session resource models.

Mirrors the test surface of ``test_memory_stores_models.py``: every
validation rule has a single test, no fixtures, no DB.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from aios.models.github_repositories import (
    MAX_REPOS_PER_SESSION,
    GithubRepositoryResource,
    GithubRepositoryResourceEcho,
    GithubRepositoryUpdate,
    validate_resources,
)


def _resource(
    *,
    url: str = "https://github.com/octocat/Hello-World",
    mount_path: str = "/workspace/repo",
    token: str = "ghp_secret",
) -> GithubRepositoryResource:
    return GithubRepositoryResource(
        type="github_repository",
        url=url,
        mount_path=mount_path,
        authorization_token=SecretStr(token),
    )


class TestGithubRepositoryResourceCreate:
    def test_minimal_valid(self) -> None:
        r = _resource()
        assert r.type == "github_repository"
        assert r.url == "https://github.com/octocat/Hello-World"
        assert r.mount_path == "/workspace/repo"
        assert r.authorization_token.get_secret_value() == "ghp_secret"

    def test_token_required(self) -> None:
        with pytest.raises(ValidationError, match="authorization_token"):
            GithubRepositoryResource(
                type="github_repository",
                url="https://github.com/octocat/Hello-World",
                mount_path="/workspace/repo",
            )  # type: ignore[call-arg]

    def test_token_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="authorization_token"):
            GithubRepositoryResource(
                type="github_repository",
                url="https://github.com/octocat/Hello-World",
                mount_path="/workspace/repo",
                authorization_token=SecretStr(""),
            )

    def test_url_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            _resource(url="")

    def test_mount_path_must_be_absolute(self) -> None:
        with pytest.raises(ValidationError):
            _resource(mount_path="repo")

    def test_mount_path_rejects_traversal(self) -> None:
        # The regex catches the '..' segment via the `[^/\x00]+` pattern
        # — `..` is two dot chars which match the segment, but the
        # post-validator catches it. Confirm both layers fire.
        with pytest.raises(ValidationError, match="traversal"):
            _resource(mount_path="/workspace/../etc")

    def test_mount_path_rejects_dot_segment(self) -> None:
        with pytest.raises(ValidationError, match="traversal"):
            _resource(mount_path="/workspace/./repo")

    def test_git_identity_fields_optional_default_none(self) -> None:
        r = _resource()
        assert r.git_user_name is None
        assert r.git_user_email is None

    def test_git_identity_fields_accepted(self) -> None:
        r = GithubRepositoryResource(
            type="github_repository",
            url="https://github.com/octocat/Hello-World",
            mount_path="/workspace/repo",
            authorization_token=SecretStr("ghp_secret"),
            git_user_name="Agent JN",
            git_user_email="agent+jn@example.com",
        )
        assert r.git_user_name == "Agent JN"
        assert r.git_user_email == "agent+jn@example.com"

    def test_mount_path_rejects_memory_overlap(self) -> None:
        with pytest.raises(ValidationError, match="/mnt/memory"):
            _resource(mount_path="/mnt/memory/foo")

    def test_mount_path_exact_memory_root_rejected(self) -> None:
        with pytest.raises(ValidationError, match="/mnt/memory"):
            _resource(mount_path="/mnt/memory")

    def test_mount_path_exact_workspace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="/workspace"):
            _resource(mount_path="/workspace")

    def test_mount_path_workspace_subdir_allowed(self) -> None:
        r = _resource(mount_path="/workspace/repo")
        assert r.mount_path == "/workspace/repo"

    def test_mount_path_with_special_chars_allowed(self) -> None:
        # Anything not /, NUL, '.', '..' is fine.
        r = _resource(mount_path="/workspace/My Repo (cloned)/v1")
        assert r.mount_path == "/workspace/My Repo (cloned)/v1"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            GithubRepositoryResource.model_validate(
                {
                    "type": "github_repository",
                    "url": "https://github.com/o/r",
                    "mount_path": "/workspace/repo",
                    "authorization_token": "ghp_x",
                    "branch": "main",  # not in our v1 schema
                }
            )


class TestGithubRepositoryUpdate:
    def test_token_required(self) -> None:
        with pytest.raises(ValidationError):
            GithubRepositoryUpdate()  # type: ignore[call-arg]

    def test_token_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="authorization_token"):
            GithubRepositoryUpdate(authorization_token=SecretStr(""))

    def test_immutable_fields_rejected(self) -> None:
        # url, mount_path are not in the update body — sending them is a 4xx.
        with pytest.raises(ValidationError, match="extra"):
            GithubRepositoryUpdate.model_validate(
                {
                    "authorization_token": "ghp_new",
                    "url": "https://github.com/other/repo",
                }
            )

    def test_git_identity_fields_optional_default_none(self) -> None:
        u = GithubRepositoryUpdate(authorization_token=SecretStr("ghp_new"))
        assert u.git_user_name is None
        assert u.git_user_email is None

    def test_git_identity_fields_accepted_alongside_token(self) -> None:
        u = GithubRepositoryUpdate(
            authorization_token=SecretStr("ghp_new"),
            git_user_name="Agent JN",
            git_user_email="agent+jn@example.com",
        )
        assert u.git_user_name == "Agent JN"
        assert u.git_user_email == "agent+jn@example.com"


class TestValidateResources:
    def test_empty_list_ok(self) -> None:
        validate_resources([])

    def test_unique_mount_paths_ok(self) -> None:
        validate_resources(
            [
                _resource(mount_path="/workspace/a"),
                _resource(mount_path="/workspace/b"),
            ]
        )

    def test_duplicate_mount_paths_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate mount_path"):
            validate_resources(
                [
                    _resource(mount_path="/workspace/repo"),
                    _resource(mount_path="/workspace/repo"),
                ]
            )

    def test_over_cap_rejected(self) -> None:
        too_many = [
            _resource(mount_path=f"/workspace/r{i}") for i in range(MAX_REPOS_PER_SESSION + 1)
        ]
        with pytest.raises(ValueError, match=str(MAX_REPOS_PER_SESSION)):
            validate_resources(too_many)


class TestGithubRepositoryResourceEcho:
    """The echo is what's serialized over the wire on read paths.

    Token is intentionally absent; ``id`` is the per-attachment ULID.
    """

    def test_no_token_field(self) -> None:
        echo_fields = set(GithubRepositoryResourceEcho.model_fields)
        assert "authorization_token" not in echo_fields
        assert {"id", "type", "url", "mount_path", "created_at", "updated_at"} <= echo_fields

    def test_default_type_literal(self) -> None:
        from datetime import UTC, datetime

        e = GithubRepositoryResourceEcho(
            id="ghrepo_01TEST",
            url="https://github.com/o/r",
            mount_path="/workspace/r",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        assert e.type == "github_repository"

    def test_echoes_git_identity_fields(self) -> None:
        from datetime import UTC, datetime

        e = GithubRepositoryResourceEcho(
            id="ghrepo_01TEST",
            url="https://github.com/o/r",
            mount_path="/workspace/r",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            git_user_name="Agent JN",
            git_user_email="agent+jn@example.com",
        )
        assert e.git_user_name == "Agent JN"
        assert e.git_user_email == "agent+jn@example.com"

    def test_git_identity_fields_default_none(self) -> None:
        from datetime import UTC, datetime

        e = GithubRepositoryResourceEcho(
            id="ghrepo_01TEST",
            url="https://github.com/o/r",
            mount_path="/workspace/r",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        assert e.git_user_name is None
        assert e.git_user_email is None
