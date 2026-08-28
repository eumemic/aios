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
# The account browser containers' bridge (jarbot#106 §6.2). ONE shared network
# for every account's computer — per-account networks would exhaust Docker's
# default address pool at ~31 — with inter-container communication OFF, so
# browser containers cannot reach each other and (via Docker's default
# inter-bridge isolation) nothing on ``aios-sandbox`` can reach them. Browser
# containers publish no ports; the worker reaches them via ``docker exec``
# only and does NOT join this network.
BROWSER_NETWORK_NAME = "aios-browser"
_BROWSER_NETWORK_ICC_OPTION = "com.docker.network.bridge.enable_icc"


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


async def ensure_browser_network() -> None:
    """Idempotently create the browser network with ICC disabled and IPv6 off.

    Same concurrent-startup race discipline as :func:`ensure_sandbox_network`
    (attempt, then re-verify the desired condition), plus TWO hard invariant
    checks — ``docker network create`` flags are INERT for a pre-existing
    network, and a live network must never be torn down and recreated, so each
    load-bearing property is re-verified on an already-existing network rather
    than assumed:

    * **ICC off** keeps one account's computer from reaching another's
      (jarbot#106 §6.2 phase gate) — a pre-existing ``aios-browser`` whose ICC
      option is not ``"false"`` hard-fails the worker.
    * **IPv6 off** is what makes the browser's egress lockdown
      (:func:`apply_browser_deny_internal`) sound: that lockdown is IPv4-only,
      so a v6 route would let untrusted web content bypass it to v6 internal /
      link-local / metadata. A pre-existing network with IPv6 enabled therefore
      hard-fails too, rather than the IPv4-only ``--ipv6=false`` create flag
      being silently trusted on a network it can no longer affect.
    """
    if not await _network_exists(BROWSER_NETWORK_NAME):
        rc, _, stderr_bytes = await run_docker_cli(
            [
                "docker",
                "network",
                "create",
                "--ipv6=false",
                "-o",
                f"{_BROWSER_NETWORK_ICC_OPTION}=false",
                BROWSER_NETWORK_NAME,
            ]
        )
        if rc == 0:
            log.info("sandbox.browser_network_created", network=BROWSER_NETWORK_NAME)
        elif not await _network_exists(BROWSER_NETWORK_NAME):
            raise RuntimeError(
                f"failed to create browser network {BROWSER_NETWORK_NAME!r}: "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )

    icc = await _network_option(BROWSER_NETWORK_NAME, _BROWSER_NETWORK_ICC_OPTION)
    if icc != "false":
        raise RuntimeError(
            f"browser network {BROWSER_NETWORK_NAME!r} has inter-container "
            f"communication enabled ({_BROWSER_NETWORK_ICC_OPTION}={icc!r}; expected "
            "'false'). Create flags are inert for an existing network: remove the "
            "network while no browser containers run and let the worker recreate it."
        )

    if await _network_enable_ipv6(BROWSER_NETWORK_NAME):
        raise RuntimeError(
            f"browser network {BROWSER_NETWORK_NAME!r} has IPv6 enabled; the "
            "browser egress lockdown is IPv4-only, so a v6 route would bypass it. "
            "Create flags are inert for an existing network: remove the network "
            "while no browser containers run and let the worker recreate it."
        )


async def _network_exists(name: str) -> bool:
    rc, _, _ = await run_docker_cli(["docker", "network", "inspect", name])
    return rc == 0


async def _network_option(network: str, option: str) -> str | None:
    """The value of a driver ``option`` on ``network``, or ``None`` if unset
    (or the network is uninspectable)."""
    rc, stdout_bytes, _ = await run_docker_cli(
        [
            "docker",
            "network",
            "inspect",
            "--format",
            f'{{{{index .Options "{option}"}}}}',
            network,
        ]
    )
    if rc != 0:
        return None
    # ``index`` on a ``map[string]string`` yields the zero value — an empty
    # string — for a missing key (contrast ``_container_on_network``'s
    # ``<nil>``, whose map holds pointers).
    out = stdout_bytes.decode("utf-8", errors="replace").strip()
    return out or None


async def _network_enable_ipv6(network: str) -> bool:
    """Whether ``network`` has IPv6 enabled (``docker network inspect``).

    ``EnableIPv6`` is a top-level network field (not a driver ``-o`` option), so
    it reads via its own ``--format`` rather than :func:`_network_option`. An
    uninspectable network reads as ``False`` — a missing network can carry no v6
    route, and the ICC/existence checks around this one already fail loudly."""
    rc, stdout_bytes, _ = await run_docker_cli(
        ["docker", "network", "inspect", "--format", "{{.EnableIPv6}}", network]
    )
    return rc == 0 and stdout_bytes.decode("utf-8", errors="replace").strip() == "true"


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
    "BROWSER_NETWORK_NAME",
    "SANDBOX_NETWORK_NAME",
    "WORKER_NETWORK_ALIAS",
    "ensure_browser_network",
    "ensure_sandbox_network",
    "is_running_in_container",
]
