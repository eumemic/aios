"""Tests for ``aios dev`` helpers.

Focus on the pure functions whose correctness underpins the bootstrap/
teardown pipeline. The CLI commands themselves hit Docker and Postgres,
so they're covered by manual E2E rather than unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.cli.commands.dev import (
    INSTANCE_ID_PATTERN,
    MAX_DB_NAME_BYTES,
    derive_instance_id,
    derive_runtime_db_url,
    parse_env_file,
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
    secrets.write_text(
        "AIOS_API_KEY=from_secrets\n"
        "AIOS_VAULT_KEY=vk_secrets\n"
        "AIOS_DB_URL=postgresql://secrets/db\n"
    )
    dotenv.write_text("AIOS_DB_URL=postgresql://dotenv/db\n")

    # Clear any inherited aios vars so only the files + our explicit env
    # participate in resolution.
    for k in list(monkeypatch._setenv.keys() if hasattr(monkeypatch, "_setenv") else []):
        monkeypatch.delenv(k, raising=False)
    for k in (
        "AIOS_API_KEY",
        "AIOS_VAULT_KEY",
        "AIOS_DB_URL",
        "AIOS_INSTANCE_ID",
        "AIOS_API_PORT",
        "AIOS_WORKSPACE_ROOT",
    ):
        monkeypatch.delenv(k, raising=False)

    s = Settings(_env_file=(str(secrets), str(dotenv)))  # type: ignore[call-arg]
    assert s.api_key.get_secret_value() == "from_secrets"
    assert s.vault_key.get_secret_value() == "vk_secrets"
    # Later file overrides earlier.
    assert s.db_url == "postgresql://dotenv/db"

    # Process env beats both files.
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://env/db")
    s2 = Settings(_env_file=(str(secrets), str(dotenv)))  # type: ignore[call-arg]
    assert s2.db_url == "postgresql://env/db"


def test_settings_instance_id_defaults_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backwards compat: no AIOS_INSTANCE_ID in env → "default", not an error."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_INSTANCE_ID", raising=False)

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.instance_id == "default"


def test_settings_instance_id_rejects_unsafe_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pattern validator rejects hand-edited .env that contains unsafe chars."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_INSTANCE_ID", "bad-dash")

    with pytest.raises(ValidationError):
        Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
