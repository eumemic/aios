"""Live-path regression coverage for keep-last-good egress DNS."""

from aios.sandbox.backends.base import CommandResult
from aios.sandbox.registry import SandboxRegistry
from tests.helpers.sandbox import FakeBackend, make_handle


def _result(stdout: str = "") -> CommandResult:
    return CommandResult(
        exit_code=0, stdout=stdout, stderr="", timed_out=False, truncated=False
    )


async def test_live_stamp_and_refresh_preserve_unresolved_host_last_good() -> None:
    """The reader-produced unresolved shape retains the live installed pin."""
    host = "unresolved.example"
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    handle = make_handle(session_id="sess_X")
    registry._handles["sess_X"] = handle
    backend.sidecar_results = [
        _result(f"{host} 9.9.9.9\n"),
        _result(
            "=filter=\n"
            "=nat=\n"
            "-A OUTPUT -d 9.9.9.9/32 -p tcp -m tcp --dport 443 -j DNAT "
            "--to-destination 172.18.0.2:49152\n"
        ),
        _result(),
    ]

    await registry._stamp_egress_state(
        handle,
        credential_hosts=frozenset({host}),
        limited_hosts=frozenset(),
        fallback_proxy_port=49152,
        runtime=None,
    )
    stamped = registry._egress_states["sess_X"]
    assert stamped.pinned == {host: {"9.9.9.9": 0}}

    await registry._refresh_egress_once()

    assert stamped.pinned == {host: {"9.9.9.9": 0}}
    assert len([d for d in backend.calls if d[1] == "run_netns_sidecar"]) == 3
