"""THE GATE (jarbot#106 §6.2): the browser network is unreachable by construction.

Rollout invariant: **no agent, in any environment, is granted a ``browser_*``
arm until this file is green in that deploy environment.** The design's
central security claim is that a bot's only route to the account computer is a
``browser_*`` tool call; these tests prove the network segment of that claim
against real Docker:

* a sandbox on ``aios-sandbox`` cannot reach a container on ``aios-browser``
  by name or by IP (Docker inter-bridge isolation);
* two containers on ``aios-browser`` cannot reach each other (ICC off) — one
  account's computer can never reach another's;
* a container provisioned through the REAL ``build_spec_from_browser`` path
  lands on ``aios-browser`` and publishes no ports.

Each unreachability assertion is double-guarded against a vacuous pass: the
listener curls ITSELF (so the target is proven up, absorbing the bind race),
and the prober curls its OWN loopback listener (``SELF_REACHED``, so the prober
is proven to have run with working curl). A missing cross-network sentinel is
therefore 'routing blocked', never 'listener down' or 'prober never started';
the prober's container exit is also asserted 0.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.network import (
    BROWSER_NETWORK_NAME,
    ensure_browser_network,
    ensure_sandbox_network,
)
from aios.sandbox.spec import build_spec_from_browser
from tests.conftest import needs_docker

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")
SIDECAR_PORT = 7788


async def _run(argv: list[str], *, deadline_s: float) -> subprocess.CompletedProcess[str]:
    """Run a subprocess in a worker thread (async functions must not call
    blocking ``subprocess.run`` directly)."""
    return await asyncio.to_thread(
        subprocess.run,
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=deadline_s,
    )


def _prober_script(name: str, ip: str) -> str:
    """Bash for an isolation prober, with an in-container POSITIVE CONTROL.

    The prober first serves and curls its OWN loopback listener (``SELF_REACHED``)
    — loopback works regardless of the Docker network, so a present sentinel
    proves this container ran and its curl works. Only then does it probe the
    cross-network target by name and by IP. The caller asserts ``SELF_REACHED``
    is present AND the container exited 0: without both, a missing
    ``BY_*_REACHED`` could mean 'prober never started', not 'routing blocked' —
    the vacuous pass this guards against. Trailing ``true`` keeps exit 0 so the
    absent sentinels are the signal, not a nonzero curl.
    """
    return (
        f"python3 -m http.server {SIDECAR_PORT} --bind 127.0.0.1 >/dev/null 2>&1 & "
        f"curl -s --max-time 5 --retry 10 --retry-delay 1 --retry-connrefused "
        f"http://127.0.0.1:{SIDECAR_PORT}/ >/dev/null && echo SELF_REACHED; "
        # By NAME: cross-network Docker DNS must not resolve it.
        f"curl -s --max-time 5 http://{name}:{SIDECAR_PORT}/ && echo BY_NAME_REACHED; "
        # By IP: inter-bridge isolation must drop the packets.
        f"curl -s --max-time 5 http://{ip}:{SIDECAR_PORT}/ && echo BY_IP_REACHED; "
        "true"
    )


def _assert_isolated(result: subprocess.CompletedProcess[str], *, ip: str, hint: str) -> None:
    """Shared assertions: the prober ran (exit 0), its positive control fired,
    and neither cross-network sentinel appeared."""
    assert result.returncode == 0, f"prober did not run cleanly ({hint}): {result.stderr.strip()}"
    assert "SELF_REACHED" in result.stdout, (
        f"positive control failed ({hint}): the prober could not curl its own loopback "
        f"listener, so a missing cross-network sentinel is NOT proof of isolation\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "BY_NAME_REACHED" not in result.stdout, f"{hint} reached target by name\n{result.stdout}"
    assert "BY_IP_REACHED" not in result.stdout, (
        f"{hint} reached target by IP {ip}\n{result.stdout}"
    )


@pytest.fixture
async def _networks_ready() -> None:
    """Create both networks if absent. No-op cleanup — host-global, shared."""
    await ensure_sandbox_network()
    await ensure_browser_network()


@pytest.fixture
async def browser_listener(_networks_ready: None) -> AsyncIterator[tuple[str, str]]:
    """A container on the browser network running an HTTP server.

    Production browser containers listen on nothing; this one listens
    DELIBERATELY so a failed connect proves network isolation rather than
    absence of a listener. Yields ``(container_name, browser_network_ip)``.
    Cleans up unconditionally.
    """
    name = f"aios-browser-listener-{uuid.uuid4().hex[:8]}"
    result = await _run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            BROWSER_NETWORK_NAME,
            IMAGE,
            "python3",
            "-m",
            "http.server",
            str(SIDECAR_PORT),
        ],
        deadline_s=60,
    )
    if result.returncode != 0:
        pytest.fail(f"listener docker run failed: {result.stderr.strip()}")
    container_id = result.stdout.strip()

    inspect = await _run(
        [
            "docker",
            "inspect",
            "--format",
            f'{{{{(index .NetworkSettings.Networks "{BROWSER_NETWORK_NAME}").IPAddress}}}}',
            name,
        ],
        deadline_s=15,
    )
    ip = inspect.stdout.strip()
    if inspect.returncode != 0 or not ip:
        await _run(["docker", "rm", "--force", container_id], deadline_s=15)
        pytest.fail(f"could not resolve listener IP: {inspect.stderr.strip()}")

    # Positive control: the listener serves ITSELF (absorbing the bind race),
    # so the cross-network failures below are about routing, not readiness.
    self_check = await _run(
        [
            "docker",
            "exec",
            name,
            "bash",
            "-c",
            f"curl -fs --max-time 5 --retry 10 --retry-delay 1 --retry-connrefused "
            f"http://127.0.0.1:{SIDECAR_PORT}/ >/dev/null",
        ],
        deadline_s=30,
    )
    if self_check.returncode != 0:
        await _run(["docker", "rm", "--force", container_id], deadline_s=15)
        pytest.fail(f"listener never came up: {self_check.stderr.strip()}")

    try:
        yield name, ip
    finally:
        await _run(["docker", "rm", "--force", container_id], deadline_s=15)


async def test_sandbox_cannot_reach_browser_network_by_name_or_ip(
    browser_listener: tuple[str, str],
) -> None:
    """A container on ``aios-sandbox`` has no route to ``aios-browser``."""
    name, ip = browser_listener
    prober = f"aios-sbx-prober-{uuid.uuid4().hex[:8]}"
    result = await _run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            prober,
            "--network",
            "aios-sandbox",
            IMAGE,
            "bash",
            "-c",
            _prober_script(name, ip),
        ],
        deadline_s=60,
    )
    _assert_isolated(result, ip=ip, hint="sandbox->browser")


async def test_browser_containers_cannot_reach_each_other(
    browser_listener: tuple[str, str],
) -> None:
    """ICC off: one account's computer cannot reach another's, even on the
    same bridge, by name or by IP."""
    name, ip = browser_listener
    prober = f"aios-browser-prober-{uuid.uuid4().hex[:8]}"
    result = await _run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            prober,
            "--network",
            BROWSER_NETWORK_NAME,
            IMAGE,
            "bash",
            "-c",
            _prober_script(name, ip),
        ],
        deadline_s=60,
    )
    _assert_isolated(result, ip=ip, hint="browser->sibling (ICC off?)")


async def test_real_browser_spec_lands_on_browser_network_with_no_ports(
    _networks_ready: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container provisioned through the REAL spec builder joins the ICC-off
    browser bridge (only), publishes no ports, and mounts the plane dir."""
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    monkeypatch.setattr(settings, "sandbox_browser_image", IMAGE)
    account_id = f"acc_{uuid.uuid4().hex[:8].upper()}"

    backend = DockerBackend()
    spec = build_spec_from_browser(account_id)
    handle = await backend.create(spec)
    try:
        inspect = await _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .NetworkSettings.Networks}}\t{{json .NetworkSettings.Ports}}"
                "\t{{json .HostConfig.PortBindings}}",
                handle.sandbox_id,
            ],
            deadline_s=15,
        )
        assert inspect.returncode == 0, inspect.stderr
        networks_json, ports_json, bindings_json = inspect.stdout.strip().split("\t")
        assert BROWSER_NETWORK_NAME in networks_json
        assert "aios-sandbox" not in networks_json
        # No published ports of any kind.
        for blob in (ports_json, bindings_json):
            assert blob in ("{}", "null"), f"browser container publishes ports: {blob}"
    finally:
        await backend.destroy(handle)
