"""Tests for ``aios dev`` helpers.

Focus on the pure functions whose correctness underpins the bootstrap/
teardown pipeline. The CLI commands themselves hit Docker and Postgres,
so they're covered by manual E2E rather than unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from aios.cli.commands import dev as dev_mod
from aios.cli.commands.dev import (
    INSTANCE_ID_PATTERN,
    MAX_DB_NAME_BYTES,
    _resolve_admin_url,
    assert_not_shared_in_worktree,
    classify_instance_mode,
    db_name_is_shared,
    derive_instance_id,
    derive_runtime_db_url,
    is_linked_worktree,
    parse_env_file,
    runtime_db_url,
    validate_instance_id_for_db,
    write_env_atomic,
)

# ── derive_instance_id ─────────────────────────────────────────────────────


def test_derive_instance_id_matches_pattern(tmp_path: Path) -> None:
    repo = tmp_path / "worktree-sleepy-forging-squirrel-wvPF4Y"
    repo.mkdir()
    result = derive_instance_id(repo)
    assert INSTANCE_ID_PATTERN.fullmatch(result) is not None


def test_derive_instance_id_same_basename_different_paths_collides_never(
    tmp_path: Path,
) -> None:
    """Two worktrees both named `aios` in different parent dirs must not collide."""
    a = tmp_path / "project-a" / "aios"
    b = tmp_path / "project-b" / "aios"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    assert derive_instance_id(a) != derive_instance_id(b)


def test_derive_instance_id_leading_digit_gets_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "9x-branch"
    repo.mkdir()
    result = derive_instance_id(repo)
    assert result.startswith("w_")
    assert INSTANCE_ID_PATTERN.fullmatch(result) is not None


def test_derive_instance_id_long_basename_truncates_and_stays_under_limit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / ("worktree-" + "x" * 200)
    repo.mkdir()
    result = derive_instance_id(repo)
    # Full DB name is "aios_dev_<id>"; must fit in 63 bytes.
    assert len(("aios_dev_" + result).encode()) <= MAX_DB_NAME_BYTES
    # And still collision-resistant — suffix is always present.
    assert "_" in result
    assert INSTANCE_ID_PATTERN.fullmatch(result) is not None


def test_derive_instance_id_non_ascii_folds_to_safe_chars(tmp_path: Path) -> None:
    repo = tmp_path / "café-branch"  # "café-branch"
    repo.mkdir()
    result = derive_instance_id(repo)
    assert INSTANCE_ID_PATTERN.fullmatch(result) is not None


def test_derive_instance_id_empty_basename_survives(tmp_path: Path) -> None:
    """Edge case: a path whose basename sanitizes to empty still produces a valid id."""
    repo = tmp_path / "---"
    repo.mkdir()
    result = derive_instance_id(repo)
    # "---" sanitizes to "" then gets the leading-digit-guard "w_" prefix.
    assert result.startswith("w_")
    assert INSTANCE_ID_PATTERN.fullmatch(result) is not None


def test_derive_instance_id_deterministic(tmp_path: Path) -> None:
    """Same path → same id across calls (important for bootstrap idempotency)."""
    repo = tmp_path / "worktree-x"
    repo.mkdir()
    assert derive_instance_id(repo) == derive_instance_id(repo)


# ── validate_instance_id_for_db ────────────────────────────────────────────


def test_validate_instance_id_accepts_safe_values() -> None:
    validate_instance_id_for_db("ok")
    validate_instance_id_for_db("a_b_123")
    validate_instance_id_for_db("_leading_underscore_ok")


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "9start",  # leading digit
        "has-dash",  # dash
        "has.dot",  # dot
        "UPPER",  # uppercase
        "has space",  # space
        'inject"; DROP TABLE--',  # SQL injection surface
    ],
)
def test_validate_instance_id_rejects_unsafe_values(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid AIOS_INSTANCE_ID"):
        validate_instance_id_for_db(bad)


def test_validate_instance_id_rejects_too_long() -> None:
    """Guard against PG's 63-byte silent truncation of identifiers."""
    # "aios_dev_" is 9 bytes; 55-char id pushes the DB name to 64 bytes.
    too_long = "a" * 55
    with pytest.raises(ValueError, match="truncates identifiers"):
        validate_instance_id_for_db(too_long)


# ── derive_runtime_db_url ──────────────────────────────────────────────────


def test_derive_runtime_db_url_swaps_only_path() -> None:
    admin = "postgresql://aios:secret@localhost:5432/postgres"
    result = derive_runtime_db_url(admin, "abcd1234")
    assert result == "postgresql://aios:secret@localhost:5432/aios_dev_abcd1234"


def test_derive_runtime_db_url_preserves_nonstandard_port() -> None:
    admin = "postgresql://user:pass@db.example.com:6543/postgres"
    result = derive_runtime_db_url(admin, "inst")
    assert result == "postgresql://user:pass@db.example.com:6543/aios_dev_inst"


def test_derive_runtime_db_url_preserves_query_string() -> None:
    admin = "postgresql://u:p@h:5432/postgres?sslmode=require"
    result = derive_runtime_db_url(admin, "inst")
    assert result == "postgresql://u:p@h:5432/aios_dev_inst?sslmode=require"


def test_derive_runtime_db_url_ipv6_host() -> None:
    admin = "postgresql://u:p@[::1]:5432/postgres"
    result = derive_runtime_db_url(admin, "inst")
    assert result == "postgresql://u:p@[::1]:5432/aios_dev_inst"


# ── parse_env_file ─────────────────────────────────────────────────────────


def test_parse_env_file_missing_returns_empty(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "does-not-exist.env") == {}


def test_parse_env_file_basic(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("# comment\nFOO=bar\n\nBAZ=qux\nQUOTED=\"with spaces\"\nSINGLEQUOTED='quoted'\n")
    assert parse_env_file(f) == {
        "FOO": "bar",
        "BAZ": "qux",
        "QUOTED": "with spaces",
        "SINGLEQUOTED": "quoted",
    }


def test_parse_env_file_ignores_malformed_lines(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("GOOD=1\nnot a key=value pair\nALSO_GOOD=2\n")
    assert parse_env_file(f) == {"GOOD": "1", "ALSO_GOOD": "2"}


# ── write_env_atomic ───────────────────────────────────────────────────────


def test_write_env_atomic_creates_new_file(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    write_env_atomic(f, {"FOO": "bar", "BAZ": "qux"})
    assert parse_env_file(f) == {"FOO": "bar", "BAZ": "qux"}


def test_write_env_atomic_preserves_unrelated_keys(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("UNRELATED=keep_me\nFOO=old\n# a comment\n")
    write_env_atomic(f, {"FOO": "new", "ADDED": "yes"})
    parsed = parse_env_file(f)
    assert parsed["UNRELATED"] == "keep_me"
    assert parsed["FOO"] == "new"
    assert parsed["ADDED"] == "yes"
    # Comment should still be present.
    assert "# a comment" in f.read_text()


def test_write_env_atomic_rewrites_existing_line_in_place(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("A=1\nFOO=old\nB=2\n")
    write_env_atomic(f, {"FOO": "new"})
    # FOO stays between A and B — line order preserved.
    lines = [line for line in f.read_text().splitlines() if line and not line.startswith("#")]
    assert lines == ["A=1", "FOO=new", "B=2"]


# ── Settings env_file layering ─────────────────────────────────────────────


def test_settings_env_file_tuple_layers_and_later_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """secrets.env provides defaults; .env overrides; process env beats both."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    dotenv = tmp_path / ".env"
    secrets.write_text("AIOS_VAULT_KEY=vk_secrets\nAIOS_DB_URL=postgresql://secrets/db\n")
    dotenv.write_text("AIOS_DB_URL=postgresql://dotenv/db\n")

    # Clear any inherited aios vars so only the files + our explicit env
    # participate in resolution.
    for k in list(monkeypatch._setenv.keys() if hasattr(monkeypatch, "_setenv") else []):
        monkeypatch.delenv(k, raising=False)
    for k in (
        "AIOS_VAULT_KEY",
        "AIOS_DB_URL",
        "AIOS_INSTANCE_ID",
        "AIOS_API_PORT",
        "AIOS_WORKSPACE_ROOT",
    ):
        monkeypatch.delenv(k, raising=False)

    s = Settings(_env_file=(str(secrets), str(dotenv)))
    assert s.vault_key.get_secret_value() == "vk_secrets"
    # Later file overrides earlier.
    assert s.db_url == "postgresql://dotenv/db"

    # Process env beats both files.
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://env/db")
    s2 = Settings(_env_file=(str(secrets), str(dotenv)))
    assert s2.db_url == "postgresql://env/db"


def test_settings_instance_id_defaults_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backwards compat: no AIOS_INSTANCE_ID in env → "default", not an error."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text(
        "AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n"
    )
    monkeypatch.delenv("AIOS_INSTANCE_ID", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.instance_id == "default"


def test_settings_instance_id_rejects_unsafe_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pattern validator rejects hand-edited .env that contains unsafe chars."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text(
        "AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n"
    )
    monkeypatch.setenv("AIOS_INSTANCE_ID", "bad-dash")

    with pytest.raises(ValidationError):
        Settings(_env_file=(str(secrets),))


# ── _resolve_admin_url (issue #824: no silent default) ─────────────────────


def test_resolve_admin_url_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var and empty secrets → None (no silent localhost:5432 default)."""
    monkeypatch.delenv("AIOS_POSTGRES_ADMIN_URL", raising=False)
    assert _resolve_admin_url({}) is None


def test_resolve_admin_url_prefers_env_over_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process env wins over the secrets.env value."""
    monkeypatch.setenv("AIOS_POSTGRES_ADMIN_URL", "postgresql://env@localhost:5433/postgres")
    secrets = {"AIOS_POSTGRES_ADMIN_URL": "postgresql://secrets@localhost:5433/postgres"}
    assert _resolve_admin_url(secrets) == "postgresql://env@localhost:5433/postgres"


def test_resolve_admin_url_uses_secrets_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no process env, the secrets.env value is returned unchanged."""
    monkeypatch.delenv("AIOS_POSTGRES_ADMIN_URL", raising=False)
    secrets = {"AIOS_POSTGRES_ADMIN_URL": "postgresql://secrets@localhost:5433/postgres"}
    assert _resolve_admin_url(secrets) == "postgresql://secrets@localhost:5433/postgres"


# ── db_name_is_shared (criterion 1) ────────────────────────────────────────


@pytest.mark.parametrize(
    "db_url,expected",
    [
        ("postgresql://h/aios", True),
        ("postgresql://localhost/test", True),
        ("postgresql://localhost:5433/aios", True),
        ("postgresql://h/aios_dev_x", False),
        ("postgresql://h/aios_dev_x?sslmode=require", False),
        ("postgresql://user:pw@host:5432/aios_dev_proj_abcd1234", False),
    ],
)
def test_db_name_is_shared(db_url: str, expected: bool) -> None:
    assert db_name_is_shared(db_url) is expected


# ── is_linked_worktree (criteria 4, 5) ─────────────────────────────────────


def test_is_linked_worktree_true_when_dirs_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter([".git/worktrees/foo\n", ".git\n"])
    monkeypatch.setattr(
        "aios.cli.commands.dev.subprocess.check_output", lambda *a, **k: next(outputs)
    )
    assert is_linked_worktree() is True


def test_is_linked_worktree_false_when_dirs_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter([".git\n", ".git\n"])
    monkeypatch.setattr(
        "aios.cli.commands.dev.subprocess.check_output", lambda *a, **k: next(outputs)
    )
    assert is_linked_worktree() is False


def test_is_linked_worktree_false_on_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr("aios.cli.commands.dev.subprocess.check_output", _boom)
    assert is_linked_worktree() is False


def test_is_linked_worktree_false_on_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr("aios.cli.commands.dev.subprocess.check_output", _boom)
    assert is_linked_worktree() is False


# ── runtime_db_url ─────────────────────────────────────────────────────────


def test_runtime_db_url_prefers_exported_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("AIOS_DB_URL=postgresql://h/aios_dev_x\n")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://h/aios")
    assert runtime_db_url(tmp_path) == "postgresql://h/aios"


def test_runtime_db_url_falls_back_to_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("AIOS_DB_URL=postgresql://h/aios_dev_x\n")
    monkeypatch.delenv("AIOS_DB_URL", raising=False)
    assert runtime_db_url(tmp_path) == "postgresql://h/aios_dev_x"


def test_runtime_db_url_none_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_DB_URL", raising=False)
    assert runtime_db_url(tmp_path) is None


# ── classify_instance_mode (criterion 7) ───────────────────────────────────

_FULL_ENV = {
    "AIOS_INSTANCE_ID": "proj_abcd1234",
    "AIOS_DB_URL": "postgresql://h/aios_dev_proj_abcd1234",
    "AIOS_API_PORT": "8090",
    "AIOS_WORKSPACE_ROOT": "/tmp/ws",
}


def test_classify_unbootstrapped_when_keys_missing() -> None:
    assert classify_instance_mode({}, None, True) == "unbootstrapped"
    partial = dict(_FULL_ENV)
    del partial["AIOS_DB_URL"]
    assert classify_instance_mode(partial, None, True) == "unbootstrapped"


def test_classify_shared_when_linked_worktree_and_shared_db() -> None:
    assert classify_instance_mode(_FULL_ENV, "postgresql://h/aios", True) == "shared"


def test_classify_isolated_when_linked_worktree_and_dev_db() -> None:
    assert (
        classify_instance_mode(_FULL_ENV, "postgresql://h/aios_dev_proj_abcd1234", True)
        == "isolated"
    )


def test_classify_isolated_when_main_checkout_even_on_shared_db() -> None:
    assert classify_instance_mode(_FULL_ENV, "postgresql://h/aios", False) == "isolated"


# ── assert_not_shared_in_worktree (criteria 2, 3) ──────────────────────────


def test_assert_raises_on_linked_shared_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr(dev_mod, "is_linked_worktree", lambda: True)
    monkeypatch.setattr(dev_mod, "get_git_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(dev_mod, "runtime_db_url", lambda _root: "postgresql://h/aios")

    messages: list[str] = []
    monkeypatch.setattr(dev_mod, "print_error", lambda msg: messages.append(msg))

    with pytest.raises(typer.Exit) as excinfo:
        assert_not_shared_in_worktree()
    assert excinfo.value.exit_code == 1
    joined = " ".join(messages)
    assert "aios dev bootstrap" in joined
    assert "AIOS_ALLOW_SHARED_DB=1" in joined


def test_assert_noop_when_not_linked_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr(dev_mod, "is_linked_worktree", lambda: False)
    monkeypatch.setattr(dev_mod, "get_git_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(dev_mod, "runtime_db_url", lambda _root: "postgresql://h/aios")
    assert_not_shared_in_worktree()  # must not raise


def test_assert_noop_when_db_is_dev_prefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr(dev_mod, "is_linked_worktree", lambda: True)
    monkeypatch.setattr(dev_mod, "get_git_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(dev_mod, "runtime_db_url", lambda _root: "postgresql://h/aios_dev_x")
    assert_not_shared_in_worktree()  # must not raise


def test_assert_noop_when_override_set_even_if_linked_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIOS_ALLOW_SHARED_DB", "1")
    monkeypatch.setattr(dev_mod, "is_linked_worktree", lambda: True)
    monkeypatch.setattr(dev_mod, "get_git_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(dev_mod, "runtime_db_url", lambda _root: "postgresql://h/aios")
    assert_not_shared_in_worktree()  # must not raise
