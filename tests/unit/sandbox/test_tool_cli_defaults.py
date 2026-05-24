"""Default-fallback behaviour for the in-sandbox ``bin/tool`` CLI.

The CLI normally reads ``TOOL_BROKER_URL`` and ``TOOL_BROKER_SECRET``
from the environment, which the worker injects at sandbox provisioning
time. Bash scripts that strip env, stale container reuse, and other
edge cases occasionally leave the CLI with empty env vars. To stay
usable, ``bin/tool`` falls back to:

* a well-known in-sandbox UDS URL for the broker, and
* a bind-mounted per-session secret file at a well-known path.

These tests pin both fallback resolvers so regressions surface as
unit-test failures rather than mysterious sandbox breakage.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest


class TestResolveBrokerUrl:
    def test_url_uses_env_when_set(
        self, tool_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TOOL_BROKER_URL", "http://foo:1234")
        assert tool_module._resolve_broker_url() == "http://foo:1234"

    def test_url_falls_back_to_uds_when_env_unset(
        self, tool_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TOOL_BROKER_URL", raising=False)
        assert tool_module._resolve_broker_url() == "unix:///var/run/aios/tool-broker.sock"

    def test_url_errors_when_env_set_empty(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Explicit-empty (e.g. ``--env TOOL_BROKER_URL=``) is an operator
        # error, not a request to use the default — surface it clearly
        # instead of producing a confusing "broker unreachable" downstream.
        monkeypatch.setenv("TOOL_BROKER_URL", "")
        with pytest.raises(SystemExit) as excinfo:
            tool_module._resolve_broker_url()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "TOOL_BROKER_URL" in err
        assert "empty" in err


class TestResolveBrokerSecret:
    def test_secret_uses_env_when_set(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("TOOL_BROKER_SECRET", "envSecret")
        # No file at the default path either.
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(tmp_path / "nonexistent"))
        assert tool_module._resolve_broker_secret() == "envSecret"

    def test_secret_falls_back_to_file(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("TOOL_BROKER_SECRET", raising=False)
        secret_file = tmp_path / "secret"
        secret_file.write_text("fileSecret")
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(secret_file))
        assert tool_module._resolve_broker_secret() == "fileSecret"

    def test_secret_strips_whitespace_from_file(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("TOOL_BROKER_SECRET", raising=False)
        secret_file = tmp_path / "secret"
        secret_file.write_text("secretWithNewline\n")
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(secret_file))
        assert tool_module._resolve_broker_secret() == "secretWithNewline"

    def test_secret_errors_when_neither_env_nor_file(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("TOOL_BROKER_SECRET", raising=False)
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(tmp_path / "no-such-file"))
        with pytest.raises(SystemExit) as excinfo:
            tool_module._resolve_broker_secret()
        assert excinfo.value.code == 2

    def test_secret_errors_when_fallback_file_empty(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # File exists but holds only whitespace — the legacy code printed
        # "no fallback at <path>" which is factually wrong (the file IS
        # there). The error must name the file and call out emptiness.
        monkeypatch.delenv("TOOL_BROKER_SECRET", raising=False)
        secret_file = tmp_path / "secret"
        secret_file.write_text("\n\n  \n")
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(secret_file))
        with pytest.raises(SystemExit) as excinfo:
            tool_module._resolve_broker_secret()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "empty" in err
        assert str(secret_file) in err

    def test_secret_errors_when_env_set_empty(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Explicit-empty is an operator error — don't silently fall back
        # to the on-disk secret file (which might exist with stale data).
        monkeypatch.setenv("TOOL_BROKER_SECRET", "")
        # Even with a fallback file present, the empty env must error.
        secret_file = tmp_path / "secret"
        secret_file.write_text("fileSecret")
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(secret_file))
        with pytest.raises(SystemExit) as excinfo:
            tool_module._resolve_broker_secret()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "TOOL_BROKER_SECRET" in err
        assert "empty" in err

    def test_secret_env_wins_over_file(
        self,
        tool_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("TOOL_BROKER_SECRET", "envSecret")
        secret_file = tmp_path / "secret"
        secret_file.write_text("fileSecret")
        monkeypatch.setattr(tool_module, "_DEFAULT_SECRET_PATH", str(secret_file))
        assert tool_module._resolve_broker_secret() == "envSecret"
