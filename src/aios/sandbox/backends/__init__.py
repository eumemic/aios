"""Sandbox backend selection.

Each backend implements :class:`aios.sandbox.backends.base.SandboxBackend`,
which is the abstract surface the rest of the sandbox subsystem talks to.
The concrete backend is chosen at worker startup via the
``AIOS_SANDBOX_BACKEND`` setting; only ``"docker"`` is recognized today.

Backends are constructed once per worker process and held by
:class:`aios.sandbox.registry.SandboxRegistry`. They are stateless from the
caller's perspective — all per-session state lives on the
:class:`SandboxHandle` the backend hands back from ``create``.
"""

from __future__ import annotations

from aios.sandbox.backends.base import (
    CommandResult,
    Disabled,
    Limited,
    ManagedSandboxRef,
    Mount,
    NetworkPolicy,
    SandboxBackend,
    SandboxBackendError,
    SandboxHandle,
    SandboxSpec,
    Unrestricted,
)


def make_backend(name: str) -> SandboxBackend:
    """Return the backend named ``name``. Raises ``ValueError`` on unknown."""
    if name == "docker":
        from aios.sandbox.backends.docker import DockerBackend

        return DockerBackend()
    raise ValueError(f"unknown sandbox backend: {name!r}. Supported: 'docker'.")


__all__ = [
    "CommandResult",
    "Disabled",
    "Limited",
    "ManagedSandboxRef",
    "Mount",
    "NetworkPolicy",
    "SandboxBackend",
    "SandboxBackendError",
    "SandboxHandle",
    "SandboxSpec",
    "Unrestricted",
    "make_backend",
]
