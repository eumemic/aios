from __future__ import annotations

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.registry import EgressRefreshState, SandboxRegistry
from aios.sandbox.setup import build_egress_refresh_script
from tests.helpers.sandbox import FakeBackend, make_handle


def test_refresh_script_adds_before_deleting_and_never_flushes() -> None:
    script = build_egress_refresh_script(
        old_ips={"api.github.com": {"1.1.1.1"}},
        new_ips={"api.github.com": {"2.2.2.2"}},
        credential_hosts={"api.github.com"},
        limited_hosts=set(),
        dnat_target=("172.18.0.2", 49152),
    )
    assert "iptables-restore" not in script
    assert "-F OUTPUT" not in script
    assert script.index("2.2.2.2") < script.index("1.1.1.1")
    assert "--dport 443 -j DNAT --to-destination 172.18.0.2:49152" in script


async def test_refresh_skips_contended_session_lock() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    registry._egress_states["sess_X"] = EgressRefreshState(
        credential_hosts=frozenset({"api.github.com"}),
        limited_hosts=frozenset(),
        networking=UnrestrictedNetworking(),
        proxy_port=49152,
        proxy_ip="172.18.0.2",
        runtime=None,
        pinned={"api.github.com": {"1.1.1.1": 0}},
    )
    lock = registry._lock_for("sess_X")
    await lock.acquire()
    try:
        await registry._refresh_egress_once()
    finally:
        lock.release()
    assert not [call for call in backend.calls if call[0] == "run_netns_sidecar"]


async def test_successful_resolves_union_then_evict_after_three_omissions() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    state = EgressRefreshState(
        credential_hosts=frozenset({"api.github.com"}),
        limited_hosts=frozenset(),
        networking=UnrestrictedNetworking(),
        proxy_port=49152,
        proxy_ip="172.18.0.2",
        runtime=None,
        pinned={"api.github.com": {"1.1.1.1": 0}},
    )
    registry._egress_states["sess_X"] = state
    for _ in range(2):
        await registry._merge_egress_resolutions("sess_X", {"api.github.com": {"2.2.2.2"}})
        assert set(state.pinned["api.github.com"]) == {"1.1.1.1", "2.2.2.2"}
    await registry._merge_egress_resolutions("sess_X", {"api.github.com": {"2.2.2.2"}})
    assert set(state.pinned["api.github.com"]) == {"2.2.2.2"}
    await registry._merge_egress_resolutions("sess_X", {"api.github.com": set()})
    assert set(state.pinned["api.github.com"]) == {"2.2.2.2"}
