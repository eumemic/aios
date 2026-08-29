"""Unit tests for the ``ssh`` built-in tool.

Drives the owner-agnostic core :func:`aios.tools.ssh._do_ssh` directly — the
same testability seam ``http_request`` exposes — injecting the server list, a
fake key resolver, and an optional suppression hook. The asyncssh seam
(``connect`` / ``import_private_key`` / ``import_known_hosts``) and the IP-pin
resolvers are patched at the module boundary, so no real network or crypto runs
(the unit socket-guard permits loopback only; these tests touch neither).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

from aios.models.agents import (
    GenericChildBinding,
    SshPermissionPolicy,
    SshServerSpec,
    StepSurface,
)
from aios.services.vaults import ResolvedSshKey
from aios.tools import ssh
from aios.tools.invoke import ToolBail

_HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyDataForTestsOnly"


def _server(
    *,
    name: str = "prod",
    host: str = "web.example.com",
    port: int = 22,
    username: str = "deploy",
    credential: str = "PROD_KEY",
    enabled: bool = True,
    permission_policy: SshPermissionPolicy | None = None,
    read_allow: bool = False,
) -> SshServerSpec:
    return SshServerSpec(
        name=name,
        host=host,
        port=port,
        username=username,
        host_keys=[_HOST_KEY],
        credential=credential,
        enabled=enabled,
        permission_policy=permission_policy,
        read_allow=read_allow,
    )


def _surface(ssh_servers: list[SshServerSpec]) -> StepSurface:
    return StepSurface(
        model="test/dummy",
        system="",
        tools=[],
        skills=[],
        mcp_servers=[],
        http_servers=[],
        ssh_servers=ssh_servers,
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        preempt_policy="wait",
        binding=GenericChildBinding(session_id="ses_test"),
    )


def _resolver(key: ResolvedSshKey | None) -> Any:
    return AsyncMock(return_value=key)


_KEY = ResolvedSshKey(secret_name="PROD_KEY", vault_id="vlt_1", private_key="PEM", passphrase=None)


class _FakeConn:
    """Minimal stand-in for an asyncssh connection: an async context manager
    with ``run`` and ``abort``."""

    def __init__(self, proc: Any) -> None:
        self._proc = proc
        self.aborted = False

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def run(self, command: str, **kwargs: Any) -> Any:
        return self._proc

    def abort(self) -> None:
        self.aborted = True


def _proc(*, exit_status: int | None, stdout: str = "", stderr: str = "") -> Any:
    return MagicMock(exit_status=exit_status, stdout=stdout, stderr=stderr)


@contextlib.contextmanager
def _patch_ssh(
    *,
    connect: Any = None,
    pinned_ip: str | None = "203.0.113.5",
    internal_ip: str | None = "203.0.113.5",
) -> Iterator[dict[str, Any]]:
    """Patch the asyncssh + IP-pin seams the core touches. ``connect`` may be an
    AsyncMock (a fake conn) or an exception to raise."""
    connect_mock = connect if connect is not None else AsyncMock()
    pinned = AsyncMock(return_value=pinned_ip)
    internal = AsyncMock(return_value=internal_ip)
    with (
        patch("aios.tools.ssh.asyncssh.connect", connect_mock),
        patch("aios.tools.ssh.asyncssh.import_private_key", MagicMock(return_value=MagicMock())),
        patch("aios.tools.ssh.asyncssh.import_public_key", MagicMock(return_value=MagicMock())),
        patch("aios.tools.ssh.asyncssh.import_known_hosts", MagicMock(return_value=MagicMock())),
        patch.object(ssh, "resolve_pinned_ip", pinned),
        patch.object(ssh, "resolve_internal_ip", internal),
    ):
        yield {"connect": connect_mock, "pinned": pinned, "internal": internal}


async def _run(servers: list[SshServerSpec], arguments: dict[str, Any], **patch_kw: Any) -> Any:
    with _patch_ssh(**patch_kw):
        return await ssh._do_ssh(servers=servers, arguments=arguments, resolve_key=_resolver(_KEY))


@pytest.mark.asyncio
async def test_unknown_server_ref_bails() -> None:
    with pytest.raises(ToolBail, match="unknown server_ref"):
        await _run([_server()], {"server_ref": "nope", "command": "ls"})


@pytest.mark.asyncio
async def test_disabled_server_is_invisible() -> None:
    with pytest.raises(ToolBail, match="unknown server_ref"):
        await _run([_server(enabled=False)], {"server_ref": "prod", "command": "ls"})


@pytest.mark.asyncio
async def test_missing_credential_bails() -> None:
    with _patch_ssh(), pytest.raises(ToolBail, match="no ssh_key credential named 'PROD_KEY'"):
        await ssh._do_ssh(
            servers=[_server()],
            arguments={"server_ref": "prod", "command": "ls"},
            resolve_key=_resolver(None),
        )


@pytest.mark.asyncio
async def test_blocked_host_bails() -> None:
    with pytest.raises(ToolBail, match="private/internal address"):
        await _run([_server()], {"server_ref": "prod", "command": "ls"}, pinned_ip=None)


@pytest.mark.asyncio
async def test_allow_internal_routes_to_internal_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    # An operator-allowlisted host uses resolve_internal_ip (which skips the
    # private-range block) rather than resolve_pinned_ip.
    from aios.config import get_settings

    monkeypatch.setattr(
        type(get_settings()),
        "ssh_allow_internal_host_set",
        property(lambda self: frozenset({"web.example.com"})),
    )
    conn = _FakeConn(_proc(exit_status=0, stdout="ok"))
    with _patch_ssh(connect=AsyncMock(return_value=conn), pinned_ip=None) as handles:
        result = await ssh._do_ssh(
            servers=[_server()],
            arguments={"server_ref": "prod", "command": "ls"},
            resolve_key=_resolver(_KEY),
        )
    handles["internal"].assert_awaited_once()
    handles["pinned"].assert_not_awaited()
    assert result == {"exit_code": 0, "stdout": "ok", "stderr": ""}


@pytest.mark.asyncio
async def test_nonzero_exit_is_success_shaped() -> None:
    conn = _FakeConn(_proc(exit_status=2, stdout="", stderr="boom"))
    result = await _run(
        [_server()],
        {"server_ref": "prod", "command": "false"},
        connect=AsyncMock(return_value=conn),
    )
    assert result == {"exit_code": 2, "stdout": "", "stderr": "boom"}


@pytest.mark.asyncio
async def test_signal_death_maps_to_minus_one() -> None:
    conn = _FakeConn(_proc(exit_status=None, stderr="Killed"))
    result = await _run(
        [_server()], {"server_ref": "prod", "command": "x"}, connect=AsyncMock(return_value=conn)
    )
    assert result["exit_code"] == -1


@pytest.mark.asyncio
async def test_streams_truncated_only_when_over_cap() -> None:
    from aios.config import get_settings

    cap = get_settings().ssh_max_output_chars
    conn = _FakeConn(_proc(exit_status=0, stdout="a" * (cap + 10), stderr="hi"))
    result = await _run(
        [_server()], {"server_ref": "prod", "command": "cat"}, connect=AsyncMock(return_value=conn)
    )
    assert result["stdout"] == "a" * cap
    assert result["stdout_truncated"] is True
    assert result["stderr"] == "hi"
    assert "stderr_truncated" not in result  # under cap → flag absent


@pytest.mark.asyncio
async def test_host_key_mismatch_bails() -> None:
    exc = asyncssh.HostKeyNotVerifiable("bad")
    with pytest.raises(ToolBail, match="host key verification failed"):
        await _run(
            [_server()],
            {"server_ref": "prod", "command": "ls"},
            connect=AsyncMock(side_effect=exc),
        )


@pytest.mark.asyncio
async def test_auth_rejected_bails() -> None:
    exc = asyncssh.PermissionDenied("no")
    with pytest.raises(ToolBail, match="authentication rejected"):
        await _run(
            [_server()],
            {"server_ref": "prod", "command": "ls"},
            connect=AsyncMock(side_effect=exc),
        )


@pytest.mark.asyncio
async def test_bad_timeout_bails() -> None:
    with pytest.raises(ToolBail, match="timeout_seconds must be a positive integer"):
        await _run(
            [_server()],
            {"server_ref": "prod", "command": "ls", "timeout_seconds": -1},
            connect=AsyncMock(return_value=_FakeConn(_proc(exit_status=0))),
        )


def test_classify_permission_matches_server() -> None:
    surface = _surface([_server(permission_policy=SshPermissionPolicy(type="always_ask"))])
    assert ssh._classify_permission({"server_ref": "prod"}, surface) == "always_ask"


def test_classify_permission_none_on_unknown() -> None:
    surface = _surface([_server()])
    assert ssh._classify_permission({"server_ref": "other"}, surface) is None
    assert ssh._classify_permission({"server_ref": 123}, surface) is None


@pytest.mark.asyncio
async def test_suppression_synthesizes_and_never_dials() -> None:
    connect = AsyncMock()
    on_suppress = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    with _patch_ssh(connect=connect):
        result = await ssh._do_ssh(
            servers=[_server()],
            arguments={"server_ref": "prod", "command": "rm -rf /srv"},
            resolve_key=_resolver(_KEY),
            on_suppress=on_suppress,
        )
    assert result == {"exit_code": 0, "stdout": "", "stderr": ""}
    on_suppress.assert_awaited_once()
    connect.assert_not_awaited()  # suppressed before the dial


def test_ssh_server_suppressed_is_default_deny() -> None:
    # The real policy: default-deny, opt-in via read_allow. Guards against an
    # inversion of `ssh_server_suppressed` (which would let every write execute
    # under outbound_suppression="on").
    from aios.models.agents import ssh_server_suppressed

    assert ssh_server_suppressed(_server(read_allow=False)) is True
    assert ssh_server_suppressed(_server(read_allow=True)) is False


@pytest.mark.asyncio
async def test_read_allow_server_passes_through_real_policy() -> None:
    # Handler-level: a read_allow server is NOT synthesized (the real
    # ssh_server_suppressed policy lets it dial).
    conn = _FakeConn(_proc(exit_status=0, stdout="live"))
    from aios.models.agents import ssh_server_suppressed
    from aios.services import outbound_suppression as osup

    async def on_suppress(server: SshServerSpec, args: dict[str, Any]) -> dict[str, Any] | None:
        if not ssh_server_suppressed(server):
            return None
        return osup.ssh_synthesized_result()

    with _patch_ssh(connect=AsyncMock(return_value=conn)):
        allowed = await ssh._do_ssh(
            servers=[_server(read_allow=True)],
            arguments={"server_ref": "prod", "command": "uptime"},
            resolve_key=_resolver(_KEY),
            on_suppress=on_suppress,
        )
    assert allowed["stdout"] == "live"  # real dial, not synthesized


@pytest.mark.asyncio
async def test_command_timeout_aborts_and_bails() -> None:
    # conn.run exceeding the deadline raises TimeoutError (indistinguishable from
    # asyncio.timeout firing); the handler must abort the connection and bail.
    conn = _FakeConn(_proc(exit_status=0))

    async def _slow_run(command: str, **kwargs: Any) -> Any:
        raise TimeoutError

    conn.run = _slow_run  # type: ignore[method-assign]
    with pytest.raises(ToolBail, match="command timed out"):
        await _run(
            [_server()],
            {"server_ref": "prod", "command": "sleep 999"},
            connect=AsyncMock(return_value=conn),
        )
    assert conn.aborted is True  # the connection was closed


@pytest.mark.asyncio
async def test_corrupt_host_key_pin_bails_loud() -> None:
    # A host_keys line that passes the light syntactic gate but whose blob is
    # unparseable must fail LOUD here (import_known_hosts would silently drop it).
    server = _server().model_copy(update={"host_keys": ["ssh-ed25519 QUJD"]})
    ok_key = MagicMock(return_value=MagicMock())
    with (
        patch.object(ssh, "resolve_pinned_ip", AsyncMock(return_value="203.0.113.5")),
        patch("aios.tools.ssh.asyncssh.import_private_key", ok_key),
        pytest.raises(ToolBail, match=r"host_keys entry .* could not be parsed"),
    ):
        await ssh._do_ssh(
            servers=[server],
            arguments={"server_ref": "prod", "command": "ls"},
            resolve_key=_resolver(_KEY),
        )


@pytest.mark.asyncio
async def test_bad_key_bail_omits_secret_material() -> None:
    # The key-parse-failure branch must not echo key material into the message.
    exc = asyncssh.KeyImportError("the private key material xyzzy is malformed")
    with (
        patch.object(ssh, "resolve_pinned_ip", AsyncMock(return_value="203.0.113.5")),
        patch("aios.tools.ssh.asyncssh.import_private_key", MagicMock(side_effect=exc)),
        pytest.raises(ToolBail) as ei,
    ):
        await ssh._do_ssh(
            servers=[_server()],
            arguments={"server_ref": "prod", "command": "ls"},
            resolve_key=_resolver(_KEY),
        )
    assert "xyzzy" not in str(ei.value)  # only type(exc).__name__ is surfaced
    assert "KeyImportError" in str(ei.value)
