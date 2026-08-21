"""Live-path regression coverage for keep-last-good egress DNS."""

from aios.sandbox.backends.base import CommandResult
from aios.sandbox.registry import SandboxRegistry
from tests.helpers.sandbox import FakeBackend, make_handle


def _result(stdout: str = "") -> CommandResult:
    return CommandResult(exit_code=0, stdout=stdout, stderr="", timed_out=False, truncated=False)


async def test_live_stamp_and_refresh_preserve_unresolved_host_last_good() -> None:
    """A partial refresh retains the unresolved host's installed rule and pin."""
    unresolved_host = "unresolved.example"
    resolved_host = "resolved.example"
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    handle = make_handle(session_id="sess_X")
    registry._handles["sess_X"] = handle
    backend.sidecar_results = [
        _result(f"{unresolved_host} 9.9.9.9\n{resolved_host} 1.1.1.1\n"),
        _result(
            "=filter=\n"
            "=nat=\n"
            "-A OUTPUT -d 9.9.9.9/32 -p tcp -m tcp --dport 443 -j DNAT "
            "--to-destination 172.18.0.2:49152\n"
            "-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 443 -j DNAT "
            "--to-destination 172.18.0.2:49152\n"
        ),
        # The refresh reader resolves one host while the other is unavailable.
        _result(f"{resolved_host} 2.2.2.2\n"),
        _result(),
    ]

    await registry._stamp_egress_state(
        handle,
        credential_hosts=frozenset({unresolved_host, resolved_host}),
        limited_hosts=frozenset(),
        fallback_proxy_port=49152,
        runtime=None,
    )
    stamped = registry._egress_states["sess_X"]
    assert stamped.pinned == {
        unresolved_host: {"9.9.9.9": 0},
        resolved_host: {"1.1.1.1": 0},
    }

    await registry._refresh_egress_once()

    assert stamped.pinned[unresolved_host] == {"9.9.9.9": 0}
    assert stamped.pinned[resolved_host] == {"1.1.1.1": 1, "2.2.2.2": 0}
    sidecar_scripts = [
        call[1]["script"] for call in backend.calls if call[0] == "run_netns_sidecar"
    ]
    assert len(sidecar_scripts) == 4
    refresh_script = sidecar_scripts[-1]
    assert "2.2.2.2" in refresh_script
    assert "9.9.9.9" not in refresh_script
