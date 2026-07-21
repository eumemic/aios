"""The Docker backend ALWAYS emits ``--security-opt seccomp=<value>`` on
create (issue #807).

Unconditional by design: a misconfiguration must never silently fall back
to Docker's built-in default profile. The value is whatever the spec carries
— a host path the docker CLI reads, or the literal ``unconfined`` for the
emergency rollback. The real docker daemon is never touched: ``run_docker_cli``
is monkeypatched to capture argv and return a successful container id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import (
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    Mount,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend


def _install_capture(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch ``run_docker_cli`` to record argv and return a fake container id."""
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    return calls


def _make_spec(*, seccomp_profile: str | None = None) -> SandboxSpec:
    """Minimal SandboxSpec; only overrides ``seccomp_profile`` when given so the
    no-override case exercises the dataclass default."""
    kwargs: dict[str, object] = {}
    if seccomp_profile is not None:
        kwargs["seccomp_profile"] = seccomp_profile
    return SandboxSpec(
        session_id="sess_seccomp",
        instance_id="inst_seccomp",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={
            MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
            INSTANCE_LABEL_KEY: "inst_seccomp",
            SESSION_LABEL_KEY: "sess_seccomp",
        },
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
        **kwargs,  # type: ignore[arg-type]
    )


def _security_opt_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "--security-opt"]


async def test_security_opt_always_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_capture(monkeypatch)
    spec = _make_spec(seccomp_profile="/app/docker/seccomp-sandbox.json")
    await DockerBackend().create(spec)

    # Order-independent: the backend emits multiple --security-opt flags
    # (no-new-privileges from #812 precedes seccomp), so scan all of them
    # rather than assuming seccomp is first — matching the sibling tests below.
    assert "seccomp=/app/docker/seccomp-sandbox.json" in _security_opt_values(calls[0])


async def test_unconfined_override_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_capture(monkeypatch)
    spec = _make_spec(seccomp_profile="unconfined")
    await DockerBackend().create(spec)

    assert "seccomp=unconfined" in _security_opt_values(calls[0])


async def test_flag_present_even_with_default_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spec built without overriding seccomp_profile still yields the flag —
    proving the backend emits it unconditionally rather than gating on a value."""
    calls = _install_capture(monkeypatch)
    spec = _make_spec()  # dataclass default
    await DockerBackend().create(spec)

    seccomp_opts = [v for v in _security_opt_values(calls[0]) if v.startswith("seccomp=")]
    assert seccomp_opts, f"no --security-opt seccomp=... in argv: {calls[0]}"
