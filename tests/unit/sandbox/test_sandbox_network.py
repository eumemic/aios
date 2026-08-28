"""Unit tests for :mod:`aios.sandbox.network`. The Docker subprocess
runner is patched so no real daemon is required."""

from __future__ import annotations

import socket
from collections.abc import Callable

import pytest

from aios.sandbox import network as sandbox_network
from aios.sandbox.network import (
    BROWSER_NETWORK_NAME,
    SANDBOX_NETWORK_NAME,
    WORKER_NETWORK_ALIAS,
    ensure_browser_network,
    ensure_sandbox_network,
)

DockerResponder = Callable[[list[str]], tuple[int, bytes, bytes]]


def install_docker_responder(
    monkeypatch: pytest.MonkeyPatch, responder: DockerResponder
) -> list[list[str]]:
    """Patch the docker subprocess runner and return a log of argv lists.

    Each docker call is recorded in the returned list (callers can
    introspect order/contents) and routed through ``responder`` for the
    return triple.
    """
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return responder(argv)

    monkeypatch.setattr(sandbox_network, "run_docker_cli", fake_run)
    return calls


# ── ensure_sandbox_network: on-host worker ───────────────────────────────────


class TestEnsureNetworkOnHost:
    """Worker running on the host (no /.dockerenv) — no self-connect."""

    @pytest.fixture(autouse=True)
    def _on_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_network, "is_running_in_container", lambda: False)

    async def test_existing_network_is_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            assert argv[:3] == ["docker", "network", "inspect"]
            return 0, b"[]", b""

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_sandbox_network()

        # Only ``inspect`` runs (no create), and no connect.
        assert calls == [["docker", "network", "inspect", SANDBOX_NETWORK_NAME]]

    async def test_missing_network_is_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"]:
                return 1, b"", b"Error: No such network"
            if argv[:3] == ["docker", "network", "create"]:
                return 0, b"netid\n", b""
            pytest.fail(f"unexpected argv: {argv}")

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_sandbox_network()
        assert [c[:3] for c in calls] == [
            ["docker", "network", "inspect"],
            ["docker", "network", "create"],
        ]

    async def test_create_passes_ipv6_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """#1207: the network is created with ``--ipv6=false`` so the IPv4-only
        egress lockdown's no-IPv6 invariant is explicit at create time rather
        than resting on the Docker default. (This is the weakest of the three
        v6-disable changes — the load-bearing protection is the per-session
        ip6tables DROP — but the flag must be present on new creates.)"""

        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"]:
                return 1, b"", b"Error: No such network"
            if argv[:3] == ["docker", "network", "create"]:
                return 0, b"netid\n", b""
            pytest.fail(f"unexpected argv: {argv}")

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_sandbox_network()
        create = next(c for c in calls if c[:3] == ["docker", "network", "create"])
        assert "--ipv6=false" in create
        # The network name is still the final positional argument.
        assert create[-1] == SANDBOX_NETWORK_NAME


# ── ensure_sandbox_network: in-container worker ──────────────────────────────


class TestEnsureNetworkInContainer:
    """Worker in container — self-connect with alias on the sandbox network."""

    @pytest.fixture(autouse=True)
    def _in_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_network, "is_running_in_container", lambda: True)
        monkeypatch.setattr(socket, "gethostname", lambda: "worker-abc")

    async def test_connects_self_with_alias_when_not_joined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"]:
                return 0, b"[]", b""  # network exists
            if argv[:2] == ["docker", "inspect"]:
                # container membership check -> not on network
                return 0, b"<nil>\n", b""
            if argv[:3] == ["docker", "network", "connect"]:
                return 0, b"", b""
            pytest.fail(f"unexpected argv: {argv}")

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_sandbox_network()
        connect_calls = [c for c in calls if c[:3] == ["docker", "network", "connect"]]
        assert connect_calls == [
            [
                "docker",
                "network",
                "connect",
                "--alias",
                WORKER_NETWORK_ALIAS,
                SANDBOX_NETWORK_NAME,
                "worker-abc",
            ]
        ]

    async def test_skip_connect_when_already_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"]:
                return 0, b"[]", b""
            if argv[:2] == ["docker", "inspect"]:
                # container is already on the network — non-empty, non-<nil>
                return 0, b"map[Endpoint:abc]\n", b""
            pytest.fail(f"unexpected argv: {argv}")

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_sandbox_network()
        assert not [c for c in calls if c[:3] == ["docker", "network", "connect"]]


# ── ensure_sandbox_network: failure modes ────────────────────────────────────


class TestEnsureNetworkFailures:
    @pytest.fixture(autouse=True)
    def _on_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_network, "is_running_in_container", lambda: False)

    async def test_create_race_resolved_by_reinspect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A concurrent worker created the network between our inspect and
        our create — the re-inspect should resolve the apparent failure."""
        inspect_calls = 0

        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            nonlocal inspect_calls
            if argv[:3] == ["docker", "network", "inspect"]:
                inspect_calls += 1
                if inspect_calls == 1:
                    return 1, b"", b"Error: No such network"
                return 0, b"[]", b""  # second inspect: now exists
            if argv[:3] == ["docker", "network", "create"]:
                return 1, b"", b"Error: network with name aios-sandbox already exists\n"
            pytest.fail(f"unexpected argv: {argv}")

        install_docker_responder(monkeypatch, responder)
        await ensure_sandbox_network()  # should not raise

    async def test_create_hard_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"]:
                return 1, b"", b"Error: No such network"
            if argv[:3] == ["docker", "network", "create"]:
                return 1, b"", b"Error: daemon unreachable\n"
            pytest.fail(f"unexpected argv: {argv}")

        install_docker_responder(monkeypatch, responder)
        with pytest.raises(RuntimeError, match="failed to create sandbox network"):
            await ensure_sandbox_network()


class TestEnsureConnectFailureInContainer:
    @pytest.fixture(autouse=True)
    def _in_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_network, "is_running_in_container", lambda: True)
        monkeypatch.setattr(socket, "gethostname", lambda: "worker-abc")

    async def test_connect_hard_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"]:
                return 0, b"[]", b""
            if argv[:2] == ["docker", "inspect"]:
                # not joined, before or after the failed connect
                return 0, b"<nil>\n", b""
            if argv[:3] == ["docker", "network", "connect"]:
                return 1, b"", b"Error: container not found\n"
            pytest.fail(f"unexpected argv: {argv}")

        install_docker_responder(monkeypatch, responder)
        with pytest.raises(RuntimeError, match="failed to join worker"):
            await ensure_sandbox_network()


# ── ensure_browser_network: ICC-off invariant (jarbot#106 §6.2) ──────────────


class TestEnsureBrowserNetwork:
    """The browser bridge must be created ICC-off, and — because create flags
    are inert for a pre-existing network — a live network whose ICC option is
    not ``"false"`` must hard-fail the worker rather than run browser
    containers on an open bridge."""

    async def test_creates_with_icc_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = {"created": False}

        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"] and "--format" not in argv:
                return (0, b"[]", b"") if state["created"] else (1, b"", b"not found\n")
            if argv[:3] == ["docker", "network", "create"]:
                state["created"] = True
                return 0, b"", b""
            if argv[:3] == ["docker", "network", "inspect"] and "--format" in argv:
                return 0, b"false\n", b""
            pytest.fail(f"unexpected argv: {argv}")

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_browser_network()

        create = next(c for c in calls if c[:3] == ["docker", "network", "create"])
        assert "-o" in create
        assert "com.docker.network.bridge.enable_icc=false" in create
        assert create[-1] == BROWSER_NETWORK_NAME

    async def test_existing_icc_off_network_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"] and "--format" in argv:
                return 0, b"false\n", b""
            if argv[:3] == ["docker", "network", "inspect"]:
                return 0, b"[]", b""
            pytest.fail(f"unexpected argv: {argv}")

        calls = install_docker_responder(monkeypatch, responder)
        await ensure_browser_network()
        assert not any(c[:3] == ["docker", "network", "create"] for c in calls)

    async def test_existing_icc_open_network_hard_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing ``aios-browser`` with ICC on (older deploy, hand-made
        network) must fail the worker loudly — the create flag cannot fix it
        and the network must never be torn down while containers run."""

        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"] and "--format" in argv:
                return 0, b"<no value>\n", b""
            if argv[:3] == ["docker", "network", "inspect"]:
                return 0, b"[]", b""
            pytest.fail(f"unexpected argv: {argv}")

        install_docker_responder(monkeypatch, responder)
        with pytest.raises(RuntimeError, match="inter-container communication enabled"):
            await ensure_browser_network()

    async def test_existing_ipv6_enabled_network_hard_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing ``aios-browser`` with IPv6 enabled must fail the worker:
        the browser egress lockdown is IPv4-only, so a v6 route would bypass it,
        and ``--ipv6=false`` is inert on an already-existing network. ICC is off
        here, so the (earlier) ICC check passes and the v6 check is what fires."""

        def responder(argv: list[str]) -> tuple[int, bytes, bytes]:
            if argv[:3] == ["docker", "network", "inspect"] and "--format" in argv:
                fmt = argv[argv.index("--format") + 1]
                if ".EnableIPv6" in fmt:
                    return 0, b"true\n", b""
                return 0, b"false\n", b""  # ICC option is 'false' — ICC check passes
            if argv[:3] == ["docker", "network", "inspect"]:
                return 0, b"[]", b""
            pytest.fail(f"unexpected argv: {argv}")

        install_docker_responder(monkeypatch, responder)
        with pytest.raises(RuntimeError, match="IPv6 enabled"):
            await ensure_browser_network()
