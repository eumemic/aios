"""Live-path regression coverage for keep-last-good egress DNS."""

from aios.sandbox.backends.base import CommandResult
from aios.sandbox.credential_dns import CREDENTIAL_SENTINEL_IP
from aios.sandbox.registry import SandboxRegistry
from tests.helpers.sandbox import FakeBackend, make_handle


def _result(stdout: str = "") -> CommandResult:
    return CommandResult(exit_code=0, stdout=stdout, stderr="", timed_out=False, truncated=False)


async def test_live_stamp_and_refresh_preserve_unresolved_host_last_good() -> None:
    """A partial DNS refresh leaves the unresolved host's live pin untouched.

    Stated over LIMITED hosts: since #2042 a credential host owns no
    per-address rule at all (interception is keyed on the name), so the
    per-host filter ACCEPT is the only per-address shape the sweep still
    installs and evicts — and therefore the only one whose keep-last-good
    behaviour is observable in the emitted script.
    """
    unresolved = "unresolved.example"
    resolved = "resolved.example"
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    handle = make_handle(session_id="sess_X")
    registry._handles["sess_X"] = handle
    backend.sidecar_results = [
        # Stamp-time attribution of the rules installed by provisioning.
        _result(f"{unresolved} 9.9.9.9\n{resolved} 1.1.1.1\n"),
        _result(
            "=filter=\n"
            "-A OUTPUT -d 9.9.9.9/32 -p tcp -m tcp --dport 443 -j ACCEPT\n"
            "-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 443 -j ACCEPT\n"
            "=nat=\n"
        ),
        # No credential hosts ⇒ the proxy endpoint comes from the alias resolve.
        _result("aios-worker 172.18.0.2\n"),
        # The next live refresh resolves one host while DNS for the other fails.
        _result(f"{resolved} 2.2.2.2\n"),
        _result(),
    ]

    await registry._stamp_egress_state(
        handle,
        credential_hosts=frozenset(),
        limited_hosts=frozenset({unresolved, resolved}),
        fallback_proxy_port=49152,
        runtime=None,
    )
    state = registry._egress_states["sess_X"]
    assert state.pinned == {
        unresolved: {"9.9.9.9": 0},
        resolved: {"1.1.1.1": 0},
    }

    await registry._refresh_egress_once()

    assert state.pinned[unresolved] == {"9.9.9.9": 0}
    assert state.pinned[resolved] == {"1.1.1.1": 1, "2.2.2.2": 0}
    refresh_script = [
        call[1]["script"] for call in backend.calls if call[0] == "run_netns_sidecar"
    ][-1]
    assert "2.2.2.2" in refresh_script
    assert "9.9.9.9" not in refresh_script


async def test_live_refresh_never_retires_the_credential_sentinel_dnat() -> None:
    """Aging may evict any pinned address EXCEPT the interception chokepoint.

    In-sandbox DNS answers every credential name with the sentinel, so the one
    provisioned sentinel DNAT is what the stamp reads back and pins. Drive the
    pin all the way to eviction: the legacy per-address delete shape is
    byte-identical to that rule, and nothing has re-added a credential DNAT
    since #2042, so emitting it once would silently retire name-based
    interception — fail-open under Unrestricted.
    """
    host = "api.example"
    real_ip = "140.82.121.6"
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    handle = make_handle(session_id="sess_Y")
    registry._handles["sess_Y"] = handle
    backend.sidecar_results = [
        # Stamp: the credential name resolves to the sentinel inside the netns.
        _result(f"{host} {CREDENTIAL_SENTINEL_IP}\n"),
        _result(
            "=filter=\n"
            "=nat=\n"
            f"-A OUTPUT -d {CREDENTIAL_SENTINEL_IP}/32 -p tcp -m tcp --dport 443 -j DNAT "
            "--to-destination 172.18.0.2:49152\n"
        ),
        # Every later tick resolves the name to a real pool address instead —
        # the only way the sentinel can fall out of the fresh set at all — so
        # its pin ages past _EGRESS_EVICT_AFTER_SUCCESSES and is evicted.
        # (Apply calls pop from the same queue; their stdout is ignored.)
        *[_result(f"{host} {real_ip}\n") for _ in range(8)],
    ]

    await registry._stamp_egress_state(
        handle,
        credential_hosts=frozenset({host}),
        limited_hosts=frozenset(),
        fallback_proxy_port=49152,
        runtime=None,
    )
    state = registry._egress_states["sess_Y"]
    assert state.pinned == {host: {CREDENTIAL_SENTINEL_IP: 0}}

    for _ in range(3):
        await registry._refresh_egress_once()

    # The pin really was evicted (so the delete pass really did consider it)...
    assert state.pinned == {host: {real_ip: 0}}
    # ...and not one emitted script ever deleted the chokepoint.
    scripts = [call[1]["script"] for call in backend.calls if call[0] == "run_netns_sidecar"]
    assert scripts, "expected refresh sidecar invocations"
    for script in scripts:
        assert "DNAT" not in script
        assert CREDENTIAL_SENTINEL_IP not in script
