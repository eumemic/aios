"""Unit tests for the config-driven sandbox backend factory.

The factory ``select_sandbox_backend`` is the single seam every alternate
``SandboxBackend`` implementation rides. These tests are deterministic and
require no Docker daemon: ``DockerBackend`` shells out to the ``docker`` CLI
lazily, so constructing it has no daemon dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str | None = None):
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    if backend is not None:
        monkeypatch.setenv("AIOS_SANDBOX_BACKEND", backend)
    return Settings(_env_file=(str(secrets),))


def test_select_sandbox_backend_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``sandbox_backend='docker'`` returns a ``DockerBackend`` whose name is 'docker'."""
    from aios.sandbox.backends import select_sandbox_backend
    from aios.sandbox.backends.docker import DockerBackend

    settings = _settings(tmp_path, monkeypatch, backend="docker")
    backend = select_sandbox_backend(settings)

    assert isinstance(backend, DockerBackend)
    assert backend.name == "docker"


def test_select_sandbox_backend_unknown_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown backend name fails hard with a ValueError naming the bad value."""
    from aios.sandbox.backends import select_sandbox_backend

    settings = _settings(tmp_path, monkeypatch, backend="bogus")

    with pytest.raises(ValueError, match="bogus"):
        select_sandbox_backend(settings)


def test_sandbox_backend_default_is_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam is invisible unless overridden: the default value is 'docker'."""
    settings = _settings(tmp_path, monkeypatch)

    assert settings.sandbox_backend == "docker"
