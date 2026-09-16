"""Worker-managed Docker network the sandbox uses to reach the worker.

Two in-netns resolution paths share one hostname (``aios-worker``): Docker's
embedded DNS when the worker is on the sandbox network, and ``/etc/hosts``
populated by ``--add-host`` when the worker runs on the host. A gVisor sandbox
has only the second: its Sentry never sees the netns netfilter rules that make
``127.0.0.11`` answer, so :func:`resolve_network_alias_ipv4` turns the
worker's alias into an address ``--add-host`` can bake in there too.

Both of those lookups happen on the worker: :func:`resolve_host_gateway` turns
the daemon's ``host-gateway`` substitution into an address the egress scripts
can bake in, so neither sidecar shape has to look that alias up inside the
netns.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from pathlib import Path

from aios.config import get_settings
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


# A name that exists only inside the throwaway probe container below, so the
# address we read back is unambiguously the one Docker substituted for
# ``host-gateway`` and not some unrelated line of the image's ``/etc/hosts``.
_HOST_GATEWAY_PROBE_ALIAS = "aios-host-gateway-probe"

_host_gateway_ip: str | None = None
_host_gateway_lock = asyncio.Lock()


async def resolve_host_gateway() -> str:
    """The IPv4 address ``host-gateway`` resolves to on THIS Docker daemon.

    When the worker runs on the HOST rather than on the sandbox network, the
    sandbox is created with ``--add-host aios-worker:host-gateway`` and reaches
    the worker through whatever address the daemon substitutes there. Two things
    need that address as a VALUE rather than as a magic word:

    * The egress lockdown resolves ``aios-worker`` to build ``$PROXY_IP``. On
      the runsc sidecar shape it cannot look the alias up at all — the exec
      chroots into the read-only operator image, whose ``/etc/hosts`` never
      carried it — and Docker does not publish ``--add-host`` entries to the
      embedded DNS, so the lookup misses and the credential-host redirect is
      never installed (aios#2410).
    * The periodic egress refresh must resolve names WITHOUT consulting the
      sandbox's own, tenant-writable ``/etc/hosts``
      (:class:`aios.sandbox.setup.ResolveScope`). A name the operator supplied
      has to come from somewhere the tenant cannot reach; this is that place.

    The value is daemon-dependent — the default bridge gateway on plain Linux
    Docker, the VM's host proxy (``192.168.65.2``) on Docker Desktop — and
    ``docker network inspect bridge`` answers only the first case. So we ask
    Docker the same question the sandbox asks: run a throwaway operator-image
    container with the same ``--add-host … :host-gateway`` and read what got
    written. ``--network none`` keeps the probe off every network; extra hosts
    are written regardless of network mode.

    Cached for the life of the worker process: the substitution is daemon
    configuration (``--host-gateway-ip``), which cannot change without a daemon
    restart. **Fails hard** — a deployment that passes ``--add-host
    …:host-gateway`` on every sandbox and cannot find out what it means is
    broken, and the alternative is a lockdown that silently installs no
    credential redirect.
    """
    global _host_gateway_ip
    async with _host_gateway_lock:
        if _host_gateway_ip is None:
            _host_gateway_ip = await _probe_host_gateway()
            log.info("sandbox.host_gateway_resolved", address=_host_gateway_ip)
        return _host_gateway_ip


async def _probe_host_gateway() -> str:
    image = get_settings().docker_image
    rc, stdout_bytes, stderr_bytes = await run_docker_cli(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--add-host",
            f"{_HOST_GATEWAY_PROBE_ALIAS}:host-gateway",
            "--entrypoint",
            "cat",
            image,
            "/etc/hosts",
        ],
        timeout_s=60.0,
    )
    if rc != 0:
        raise RuntimeError(
            f"host-gateway probe failed (image {image!r}, exit {rc}): "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )
    for raw in stdout_bytes.decode("utf-8", errors="replace").splitlines():
        fields = raw.split("#", 1)[0].split()
        if len(fields) >= 2 and _HOST_GATEWAY_PROBE_ALIAS in fields[1:]:
            # Raises ValueError on a non-dotted-quad, which is the right
            # outcome: every rule this address feeds is IPv4-only.
            return str(ipaddress.IPv4Address(fields[0]))
    raise RuntimeError(
        f"host-gateway probe wrote no {_HOST_GATEWAY_PROBE_ALIAS!r} entry; this "
        "Docker daemon does not support --add-host <name>:host-gateway"
    )


async def resolve_network_alias_ipv4(alias: str, network: str = SANDBOX_NETWORK_NAME) -> str | None:
    """The IPv4 address publishing ``alias`` on ``network``, or ``None``.

    Resolved HERE, on the worker, by reading the endpoints Docker itself
    recorded — never by asking a resolver inside the sandbox netns.

    This exists for runsc. Docker publishes a network alias only through its
    embedded DNS server, which libnetwork binds on ``127.0.0.11`` **inside the
    container's netns** and reaches through netfilter rules installed in that
    same netns. A gVisor sandbox terminates its own traffic in the Sentry's
    netstack and never replays those rules (google/gvisor#7469), so the alias
    is unresolvable from inside a runsc sandbox no matter what its
    ``resolv.conf`` says. ``--dns`` does not move that: on a user-defined
    network Docker always writes ``127.0.0.11`` as the container's only
    nameserver and treats ``--dns`` as the embedded resolver's *upstream*
    forwarder. The address is therefore baked into ``/etc/hosts`` at create
    (``--add-host``), which libc consults before DNS and which runsc passes
    through as the ordinary bind mount it is — the same hosts-first shape the
    worker-on-host path already uses (:func:`resolve_host_gateway`) and the
    egress scripts already resolve through (``aios.sandbox.setup``).

    ``None`` means no running container on ``network`` claims ``alias`` — the
    caller keeps the DNS-only behavior rather than inventing an address.
    """
    rc, stdout_bytes, stderr_bytes = await run_docker_cli(
        ["docker", "ps", "--quiet", "--no-trunc", "--filter", f"network={network}"]
    )
    if rc != 0:
        raise RuntimeError(
            f"listing containers on network {network!r} failed (exit {rc}): "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )
    container_ids = stdout_bytes.decode("utf-8", errors="replace").split()
    if not container_ids:
        return None

    # ``{{json .NetworkSettings.Networks}}`` — the whole endpoint map as JSON,
    # not per-field templating: Docker's inspect templates run under
    # ``missingkey=error``, so naming a key a record does not carry kills the
    # template for that container and silently drops it from stdout. A batch
    # containing a container that exited between the two calls exits nonzero
    # but still writes the lines it did produce, so parse what came back.
    rc, stdout_bytes, _ = await run_docker_cli(
        ["docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", *container_ids]
    )
    for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
        try:
            networks = json.loads(line)
        except ValueError:
            continue
        if not isinstance(networks, dict):
            continue
        endpoint = networks.get(network)
        if not isinstance(endpoint, dict):
            continue
        # ``Aliases`` is the classic field; ``DNSNames`` is what Engine 25+
        # records. Either one is how Docker's own resolver answers the name.
        names = [*(endpoint.get("Aliases") or []), *(endpoint.get("DNSNames") or [])]
        if alias not in names:
            continue
        try:
            address = str(ipaddress.IPv4Address(str(endpoint.get("IPAddress") or "")))
        except ValueError:
            continue
        log.info("sandbox.network_alias_resolved", network=network, alias=alias, address=address)
        return address
    return None


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
    "resolve_host_gateway",
    "resolve_network_alias_ipv4",
]
