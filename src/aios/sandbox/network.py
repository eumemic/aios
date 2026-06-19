"""Worker-managed Docker network the sandbox uses to reach the worker.

Two resolution paths share one hostname (``aios-worker``): Docker's
embedded DNS when the worker is on the sandbox network, ``/etc/hosts``
populated by ``--add-host`` when the worker runs on the host.
"""

from __future__ import annotations

import socket
from pathlib import Path

from aios.logging import get_logger
from aios.sandbox._subprocess import run_docker_cli

log = get_logger("aios.sandbox.network")

SANDBOX_NETWORK_NAME = "aios-sandbox"
WORKER_NETWORK_ALIAS = "aios-worker"


def is_running_in_container() -> bool:
    """``True`` when ``/.dockerenv`` exists."""
    return Path("/.dockerenv").exists()


async def ensure_sandbox_network() -> None:
    """Idempotently create the sandbox network; if in-container, join it
    under :data:`WORKER_NETWORK_ALIAS`.

    Safe under concurrent-startup races: a failed create or connect is
    re-checked against the live state, and treated as success if the
    desired condition now holds. Other failures raise.

    Self-identification uses :func:`socket.gethostname`, which equals the
    Docker container name in Coolify and docker-compose. Deployments
    that split ``--hostname`` from ``--name`` will fail here.
    """
    if not await _network_exists(SANDBOX_NETWORK_NAME):
        # ``--ipv6=false`` makes the IPv4-only egress lockdown's no-IPv6
        # invariant explicit at create time rather than relying on the Docker
        # default (#1207). NOTE this is the WEAKEST of the v6-disable changes:
        # it is redundant against the current Docker default, does NOT defend a
        # daemon configured with default-IPv6-on, and is INERT for an
        # already-running network (which constraint #4 forbids us from
        # recreating). The load-bearing protection is the per-session
        # ``ip6tables -P OUTPUT DROP`` applied in the lockdown sidecar (see
        # ``setup.build_iptables_script``); this flag is belt-and-suspenders on
        # top of it, NOT a substitute, and must never be "fixed" by tearing
        # down and recreating the live prod network.
        rc, _, stderr_bytes = await run_docker_cli(
            ["docker", "network", "create", "--ipv6=false", SANDBOX_NETWORK_NAME]
        )
        if rc == 0:
            log.info("sandbox.network_created", network=SANDBOX_NETWORK_NAME)
        elif not await _network_exists(SANDBOX_NETWORK_NAME):
            raise RuntimeError(
                "failed to create sandbox network "
                f"{SANDBOX_NETWORK_NAME!r}: "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )

    if not is_running_in_container():
        log.info(
            "sandbox.network_worker_on_host",
            network=SANDBOX_NETWORK_NAME,
            alias=WORKER_NETWORK_ALIAS,
        )
        return

    hostname = socket.gethostname()
    if await _container_on_network(hostname, SANDBOX_NETWORK_NAME):
        log.info(
            "sandbox.network_worker_already_joined",
            network=SANDBOX_NETWORK_NAME,
            alias=WORKER_NETWORK_ALIAS,
            hostname=hostname,
        )
        return

    rc, _, stderr_bytes = await run_docker_cli(
        [
            "docker",
            "network",
            "connect",
            "--alias",
            WORKER_NETWORK_ALIAS,
            SANDBOX_NETWORK_NAME,
            hostname,
        ]
    )
    if rc != 0 and not await _container_on_network(hostname, SANDBOX_NETWORK_NAME):
        raise RuntimeError(
            f"failed to join worker {hostname!r} to sandbox network "
            f"{SANDBOX_NETWORK_NAME!r}: "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )
    log.info(
        "sandbox.network_worker_joined",
        network=SANDBOX_NETWORK_NAME,
        alias=WORKER_NETWORK_ALIAS,
        hostname=hostname,
    )


async def _network_exists(name: str) -> bool:
    rc, _, _ = await run_docker_cli(["docker", "network", "inspect", name])
    return rc == 0


async def _container_on_network(container: str, network: str) -> bool:
    # Network name lands inside a Go ``index`` template; that wants a
    # double-quoted string, not Python's single-quoted ``repr``.
    rc, stdout_bytes, _ = await run_docker_cli(
        [
            "docker",
            "inspect",
            "--format",
            f'{{{{index .NetworkSettings.Networks "{network}"}}}}',
            container,
        ]
    )
    if rc != 0:
        return False
    # ``docker inspect --format`` prints ``<nil>`` when the network key
    # is absent; a non-empty, non-``<nil>`` value means joined.
    out = stdout_bytes.decode("utf-8", errors="replace").strip()
    return bool(out) and out != "<nil>"


__all__ = [
    "SANDBOX_NETWORK_NAME",
    "WORKER_NETWORK_ALIAS",
    "ensure_sandbox_network",
    "is_running_in_container",
]
