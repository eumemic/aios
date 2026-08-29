"""End-to-end-ish test of the ssh tool against a real in-process asyncssh
server on loopback.

Unlike ``test_ssh.py`` (which mocks the asyncssh seam), this exercises the parts
mocks can't: ``import_private_key`` on a real PEM, ``import_known_hosts`` with the
``*`` pattern actually verifying the presented host key, and ``connect``/``run``
against a live server. The server is bound on 127.0.0.1 with an ephemeral port and dialed by IP;
the tool reaches it via the operator internal-host allowlist (which routes to
``resolve_internal_ip`` — loopback is otherwise blocked), so the override path is
exercised for real. Precedent shape: ``test_git_proxy.py`` (in-process server, no
docker, no real remote).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import asyncssh
import pytest

from aios.config import get_settings
from aios.services.vaults import ResolvedSshKey
from aios.tools import ssh
from aios.tools.invoke import ToolBail
from tests.unit.test_ssh import _server


class _Server(asyncssh.SSHServer):
    def begin_auth(self, username: str) -> bool:
        return True  # public-key auth is validated by the connection's authorized_keys

    def public_key_auth_supported(self) -> bool:
        return True


async def _handle(process: Any) -> None:
    # Echo the command back with a fixed exit code so the test can assert both.
    command = process.command or ""
    if command == "exit-3":
        process.exit(3)
    else:
        process.stdout.write(f"ran: {command}\n")
        process.exit(0)


@pytest.fixture
async def sshd() -> Any:
    """A loopback asyncssh server. Yields (host_pub_line, client_priv_pem, port)."""
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    client_key = asyncssh.generate_private_key("ssh-ed25519")
    server = await asyncssh.create_server(
        _Server,
        "127.0.0.1",
        0,
        server_host_keys=[host_key],
        authorized_client_keys=_authorized(client_key),
        process_factory=_handle,
    )
    port = server.sockets[0].getsockname()[1]
    host_pub_line = host_key.export_public_key().decode().strip()
    client_priv_pem = client_key.export_private_key().decode()
    try:
        yield host_pub_line, client_priv_pem, port
    finally:
        server.close()
        await server.wait_closed()


def _authorized(client_key: Any) -> Any:
    pub = client_key.export_public_key().decode().strip()
    return asyncssh.import_authorized_keys(pub)


def _allow_loopback(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    monkeypatch.setattr(
        type(get_settings()),
        "ssh_allow_internal_host_set",
        property(lambda self: frozenset({"127.0.0.1", f"127.0.0.1:{port}"})),
    )


@pytest.mark.asyncio
async def test_real_exec_success(sshd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    host_pub, client_pem, port = sshd
    _allow_loopback(monkeypatch, port)
    server = _server(host="127.0.0.1", port=port, username="tester").model_copy(
        update={"host_keys": [host_pub]}
    )
    result = await ssh._do_ssh(
        servers=[server],
        arguments={"server_ref": "prod", "command": "hello"},
        resolve_key=AsyncMock(
            return_value=ResolvedSshKey(
                secret_name="PROD_KEY", vault_id="v", private_key=client_pem
            )
        ),
    )
    assert result["exit_code"] == 0
    assert "ran: hello" in result["stdout"]


@pytest.mark.asyncio
async def test_real_nonzero_exit(sshd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    host_pub, client_pem, port = sshd
    _allow_loopback(monkeypatch, port)
    server = _server(host="127.0.0.1", port=port, username="tester").model_copy(
        update={"host_keys": [host_pub]}
    )
    result = await ssh._do_ssh(
        servers=[server],
        arguments={"server_ref": "prod", "command": "exit-3"},
        resolve_key=AsyncMock(
            return_value=ResolvedSshKey(
                secret_name="PROD_KEY", vault_id="v", private_key=client_pem
            )
        ),
    )
    assert result["exit_code"] == 3


@pytest.mark.asyncio
async def test_host_key_mismatch_refused(sshd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _host_pub, client_pem, port = sshd
    _allow_loopback(monkeypatch, port)
    # Pin a DIFFERENT host key than the server actually presents.
    wrong_pub = asyncssh.generate_private_key("ssh-ed25519").export_public_key().decode().strip()
    server = _server(host="127.0.0.1", port=port, username="tester").model_copy(
        update={"host_keys": [wrong_pub]}
    )
    with pytest.raises(ToolBail, match="host key verification failed"):
        await ssh._do_ssh(
            servers=[server],
            arguments={"server_ref": "prod", "command": "hello"},
            resolve_key=AsyncMock(
                return_value=ResolvedSshKey(
                    secret_name="PROD_KEY", vault_id="v", private_key=client_pem
                )
            ),
        )


@pytest.mark.asyncio
async def test_wrong_client_key_rejected(sshd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    host_pub, _client_pem, port = sshd
    _allow_loopback(monkeypatch, port)
    # A client key the server does not authorize.
    other_pem = asyncssh.generate_private_key("ssh-ed25519").export_private_key().decode()
    server = _server(host="127.0.0.1", port=port, username="tester").model_copy(
        update={"host_keys": [host_pub]}
    )
    with pytest.raises(ToolBail, match="authentication rejected"):
        await ssh._do_ssh(
            servers=[server],
            arguments={"server_ref": "prod", "command": "hello"},
            resolve_key=AsyncMock(
                return_value=ResolvedSshKey(
                    secret_name="PROD_KEY", vault_id="v", private_key=other_pem
                )
            ),
        )
