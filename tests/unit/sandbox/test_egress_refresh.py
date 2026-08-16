"""Unit tests for the periodic sandbox egress-refresh sweep (#1946).

Drives the real :class:`SandboxRegistry` refresh machinery against a
:class:`FakeBackend`, asserting on the exact sidecar scripts the sweep
emits — the scripts ARE the effector, so a wrong old→new delta or a
non-idempotent rule op is only visible there.
"""

from __future__ import annotations

from structlog.testing import capture_logs

from aios.sandbox.backends.base import CommandResult
from aios.sandbox.registry import (
    _EGRESS_EVICT_AFTER_SUCCESSES as _EVICT_AFTER,
)
from aios.sandbox.registry import (
    EgressRefreshState,
    SandboxRegistry,
)
from aios.sandbox.setup import build_egress_refresh_script, egress_unread_hosts
from tests.helpers.sandbox import FakeBackend, make_handle

_PROXY = ("172.18.0.2", 49152)


def _state(
    *,
    credential_hosts: frozenset[str] = frozenset({"api.github.com"}),
    limited_hosts: frozenset[str] = frozenset(),
    pinned: dict[str, dict[str, int]],
) -> EgressRefreshState:
    return EgressRefreshState(
        credential_hosts=credential_hosts,
        limited_hosts=limited_hosts,
        proxy_port=_PROXY[1],
        proxy_ip=_PROXY[0],
        runtime=None,
        pinned=pinned,
    )


def _sidecar_scripts(backend: FakeBackend) -> list[str]:
    return [call[1]["script"] for call in backend.calls if call[0] == "run_netns_sidecar"]


def _sidecar_result(exit_code: int = 0, stdout: str = "") -> CommandResult:
    return CommandResult(
        exit_code=exit_code, stdout=stdout, stderr="", timed_out=False, truncated=False
    )


def test_refresh_script_adds_before_deleting_and_never_flushes() -> None:
    script = build_egress_refresh_script(
        old_ips={"api.github.com": {"1.1.1.1"}},
        new_ips={"api.github.com": {"2.2.2.2"}},
        credential_hosts={"api.github.com"},
        limited_hosts=set(),
        dnat_target=_PROXY,
    )
    assert "iptables-restore" not in script
    assert "-F OUTPUT" not in script
    assert script.index("2.2.2.2") < script.index("1.1.1.1")
    assert "--dport 443 -j DNAT --to-destination 172.18.0.2:49152" in script


def test_refresh_script_keeps_rules_pinned_by_another_host() -> None:
    """A host eviction must not delete a category rule another host still owns."""
    old = {
        "a.example": {"1.1.1.1"},
        "b.example": {"1.1.1.1"},
    }
    script = build_egress_refresh_script(
        old_ips=old,
        new_ips={"a.example": {"2.2.2.2"}, "b.example": {"1.1.1.1"}},
        credential_hosts={"a.example", "b.example"},
        limited_hosts={"a.example", "b.example"},
        dnat_target=_PROXY,
    )

    assert "-A OUTPUT -d 2.2.2.2" in script
    assert "-D OUTPUT -d 1.1.1.1" not in script


def test_builder_refuses_deletes_when_inventory_omits_an_in_scope_host() -> None:
    """BUILDER CONTRACT ONLY — not a live-system property.

    Titled for what it actually verifies. It hand-builds an inventory that
    OMITS an in-scope host and asserts the builder emits no deletes. Today's
    sole in-tree caller **cannot produce this input**: keep-last-good in
    ``_merge_egress_resolutions`` carries an unread host forward at its old
    pins, so the host arrives PRESENT (measured — see ``egress_unread_hosts``).
    The live protection is upstream; this pins the public helper's contract,
    which otherwise turns a missing key into a delete of a live rule.
    """
    script = build_egress_refresh_script(
        old_ips={"a.example": {"1.1.1.1"}, "b.example": {"9.9.9.9"}},
        new_ips={"a.example": {"1.1.1.1", "2.2.2.2"}},  # b.example: read FAILED
        credential_hosts={"a.example", "b.example"},
        limited_hosts={"a.example", "b.example"},
        dnat_target=_PROXY,
    )

    # No deletion may be derived from a partial inventory.
    assert "-D OUTPUT" not in script
    assert "9.9.9.9" not in script
    # The refusal is surfaced, not silent.
    assert "b.example" in script
    # Adds are unaffected — a read host's new IP is still installed.
    assert '"$IPT" -A OUTPUT -d 2.2.2.2 -p tcp --dport 80 -j ACCEPT' in script


def test_builder_still_deletes_when_inventory_shows_host_owns_nothing() -> None:
    """BUILDER CONTRACT ONLY — the over-correction guard.

    Present-with-an-empty-set is AUTHORITATIVE ("read, owns nothing") and must
    still delete, or the refusal degrades into never deleting — a refusal with
    no escalation path is a different outage, not a fixed one."""
    script = build_egress_refresh_script(
        old_ips={"a.example": {"1.1.1.1"}, "b.example": {"9.9.9.9"}},
        # b.example WAS read and genuinely resolves to nothing.
        new_ips={"a.example": {"1.1.1.1"}, "b.example": set()},
        credential_hosts={"a.example", "b.example"},
        limited_hosts={"a.example", "b.example"},
        dnat_target=_PROXY,
    )

    assert (
        '"$IPT" -t nat -D OUTPUT -d 9.9.9.9 -p tcp --dport 443 '
        f"-j DNAT --to-destination {_PROXY[0]}:{_PROXY[1]} 2>/dev/null || true" in script
    )
    assert '"$IPT" -D OUTPUT -d 9.9.9.9 -p tcp --dport 80 -j ACCEPT 2>/dev/null || true' in script
    assert '"$IPT" -D OUTPUT -d 9.9.9.9 -p tcp --dport 443 -j ACCEPT 2>/dev/null || true' in script
    # Still-pinned IP is untouched.
    assert "-D OUTPUT -d 1.1.1.1" not in script


def test_refresh_script_ops_are_idempotent() -> None:
    """Every add is -C-guarded and every delete tolerates absence, so a
    retried old→new delta neither aborts under ``set -e`` nor accumulates
    duplicate rules."""
    script = build_egress_refresh_script(
        old_ips={"api.github.com": {"1.1.1.1"}},
        new_ips={"api.github.com": {"2.2.2.2"}},
        credential_hosts={"api.github.com"},
        limited_hosts={"api.github.com"},
        dnat_target=_PROXY,
    )
    for line in script.splitlines():
        if " -A OUTPUT " in line:
            guard = line.split(" 2>/dev/null || ")[0]
            assert guard == line.split(" 2>/dev/null || ")[1].replace(
                " -A OUTPUT ", " -C OUTPUT ", 1
            ), line
        if " -D OUTPUT " in line:
            assert line.endswith("2>/dev/null || true"), line


async def test_refresh_skips_contended_session_lock() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    registry._egress_states["sess_X"] = _state(pinned={"api.github.com": {"1.1.1.1": 0}})
    lock = registry._lock_for("sess_X")
    await lock.acquire()
    try:
        await registry._refresh_egress_once()
    finally:
        lock.release()
    assert not _sidecar_scripts(backend)


async def test_successful_resolves_union_then_evict_after_three_omissions() -> None:
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    state = _state(pinned={"api.github.com": {"1.1.1.1": 0}})
    registry._egress_states["sess_X"] = state
    for _ in range(2):
        await registry._merge_egress_resolutions("sess_X", {"api.github.com": {"2.2.2.2"}})
        assert set(state.pinned["api.github.com"]) == {"1.1.1.1", "2.2.2.2"}
    await registry._merge_egress_resolutions("sess_X", {"api.github.com": {"2.2.2.2"}})
    assert set(state.pinned["api.github.com"]) == {"2.2.2.2"}
    await registry._merge_egress_resolutions("sess_X", {"api.github.com": set()})
    assert set(state.pinned["api.github.com"]) == {"2.2.2.2"}


async def test_resolve_miss_retains_last_good_and_skips_refresh_sidecar() -> None:
    """A resolve miss (empty stdout) must carry the last-good pins: no refresh
    sidecar runs and the DNAT rules/state stay exactly as they were —
    otherwise a transient DNS outage would evict the placeholder's only
    route."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    state = _state(pinned={"api.github.com": {"1.1.1.1": 0}})
    registry._egress_states["sess_X"] = state

    await registry._refresh_egress_once()

    scripts = _sidecar_scripts(backend)
    # Exactly ONE sidecar ran — the resolve itself. No refresh script was
    # dispatched, so the installed rules are untouched.
    assert len(scripts) == 1
    assert "resolve_ipv4" in scripts[0]
    assert state.pinned == {"api.github.com": {"1.1.1.1": 0}}


async def test_blackhole_recovery_swaps_dnat_and_limited_accept_rules() -> None:
    """End-to-end #1946 recovery with a NON-EMPTY ``limited_hosts``: DNS
    rotates 1.1.1.1 → 2.2.2.2. The sweep must first install the new IP's
    rules (recovering the placeholder swap), then — after the eviction
    window — emit the stale IP's ``-D`` for BOTH the nat DNAT and the
    filter ACCEPTs."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    host = "api.github.com"
    state = _state(
        credential_hosts=frozenset({host}),
        limited_hosts=frozenset({host}),
        pinned={host: {"1.1.1.1": 0}},
    )
    registry._egress_states["sess_X"] = state

    # Tick 1: the new IP appears → its rules are installed immediately.
    await registry._merge_egress_resolutions("sess_X", {host: {"2.2.2.2"}})
    add_script = _sidecar_scripts(backend)[-1]
    assert (
        '"$IPT" -t nat -A OUTPUT -d 2.2.2.2 -p tcp --dport 443 '
        f"-j DNAT --to-destination {_PROXY[0]}:{_PROXY[1]}" in add_script
    )
    assert '"$IPT" -A OUTPUT -d 2.2.2.2 -p tcp --dport 80 -j ACCEPT' in add_script
    assert '"$IPT" -A OUTPUT -d 2.2.2.2 -p tcp --dport 443 -j ACCEPT' in add_script
    assert "-D OUTPUT" not in add_script  # stale IP survives the union window

    # Ticks 2-3: the stale IP stays omitted → evicted, deletes emitted for
    # both rule shapes.
    await registry._merge_egress_resolutions("sess_X", {host: {"2.2.2.2"}})
    await registry._merge_egress_resolutions("sess_X", {host: {"2.2.2.2"}})
    delete_script = _sidecar_scripts(backend)[-1]
    assert (
        '"$IPT" -t nat -D OUTPUT -d 1.1.1.1 -p tcp --dport 443 '
        f"-j DNAT --to-destination {_PROXY[0]}:{_PROXY[1]} 2>/dev/null || true" in delete_script
    )
    assert (
        '"$IPT" -D OUTPUT -d 1.1.1.1 -p tcp --dport 80 -j ACCEPT 2>/dev/null || true'
        in delete_script
    )
    assert (
        '"$IPT" -D OUTPUT -d 1.1.1.1 -p tcp --dport 443 -j ACCEPT 2>/dev/null || true'
        in delete_script
    )
    assert state.pinned == {host: {"2.2.2.2": 0}}


async def test_run_release_pops_egress_state() -> None:
    """A credentialed run sandbox's refresh stamp must die with the run —
    otherwise the 30s sweep lock-cycles the ghost entry forever and the
    dict grows unbounded under run churn."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    run_id = "wfr_01TEST"
    registry._handles[run_id] = make_handle(session_id=run_id)
    registry._egress_states[run_id] = _state(pinned={"api.github.com": {"1.1.1.1": 0}})

    await registry.release_run(run_id)

    assert run_id not in registry._egress_states
    # The uncached branch pops the stamp too (belt for the recycle path).
    registry._egress_states[run_id] = _state(pinned={"api.github.com": {"1.1.1.1": 0}})
    await registry.release_run(run_id)
    assert run_id not in registry._egress_states


async def test_failed_refresh_keeps_pinned_and_retry_converges_without_duplicates() -> None:
    """A refresh sidecar failure must NOT advance ``pinned`` (loud retry next
    tick), and the retried delta must be byte-identical — the -C add guards
    and ``|| true`` deletes make the retry idempotent instead of
    accumulating duplicate rules."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    host = "api.github.com"
    state = _state(pinned={host: {"1.1.1.1": _EVICT_AFTER - 1}})
    registry._egress_states["sess_X"] = state

    backend.sidecar_results = [_sidecar_result(exit_code=1)]
    await registry._merge_egress_resolutions("sess_X", {host: {"2.2.2.2"}})
    assert state.pinned == {host: {"1.1.1.1": _EVICT_AFTER - 1}}  # not advanced

    # Retry tick: same resolution → the SAME old→new delta is re-dispatched.
    await registry._merge_egress_resolutions("sess_X", {host: {"2.2.2.2"}})
    first, retry = _sidecar_scripts(backend)
    assert retry == first
    # The retried adds are -C-guarded (no duplicate accumulation) and the
    # retried delete tolerates the rule already being gone.
    assert (
        '"$IPT" -t nat -C OUTPUT -d 2.2.2.2 -p tcp --dport 443 '
        f"-j DNAT --to-destination {_PROXY[0]}:{_PROXY[1]} 2>/dev/null || " in retry
    )
    assert "-D OUTPUT -d 1.1.1.1" in retry
    assert retry.count("|| true") == retry.count("-D OUTPUT")
    assert state.pinned == {host: {"2.2.2.2": 0}}  # advanced on success


async def test_merge_passes_exact_old_new_sets_to_refresh_script() -> None:
    """The merge→builder wiring: the sidecar script must be byte-identical to
    ``build_egress_refresh_script`` called with the state's own params and
    the exact old/new pin sets the merge computed."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    host = "api.github.com"
    state = _state(
        credential_hosts=frozenset({host}),
        limited_hosts=frozenset({host}),
        pinned={host: {"1.1.1.1": _EVICT_AFTER - 1, "3.3.3.3": 0}},
    )
    registry._egress_states["sess_X"] = state

    await registry._merge_egress_resolutions("sess_X", {host: {"2.2.2.2", "3.3.3.3"}})

    assert _sidecar_scripts(backend)[-1] == build_egress_refresh_script(
        old_ips={host: {"1.1.1.1", "3.3.3.3"}},
        new_ips={host: {"2.2.2.2", "3.3.3.3"}},
        credential_hosts={host},
        limited_hosts={host},
        dnat_target=_PROXY,
    )


def test_pinned_seeded_from_installed_rules_not_second_resolve() -> None:
    """``pinned`` seeding reflects the rules ACTUALLY installed, not the
    stamp-time resolve: an installed-but-unresolved IP is tracked (so
    eviction can remove it) while a resolved-but-uninstalled IP is NOT
    seeded (the first sweep tick installs it through the refresh script)."""
    pinned = SandboxRegistry._seed_pinned_from_installed(
        # The stamp-time (second) resolve diverged: DNS now says 2.2.2.2 …
        {"api.github.com": {"2.2.2.2"}},
        credential_hosts=frozenset({"api.github.com"}),
        limited_hosts=frozenset(),
        # … but the apply's own in-script resolve installed 1.1.1.1.
        dnat_ips={"1.1.1.1"},
        accept_ips=set(),
    )
    assert pinned == {"api.github.com": {"1.1.1.1": 0}}


async def test_stamp_seeds_pinned_and_proxy_target_from_live_rules() -> None:
    """End-to-end stamp: resolve says 2.2.2.2, the live table says 1.1.1.1 —
    the stamped state must track the installed rule (and take the DNAT
    ``--to-destination`` from the live rule, not another resolve), so the
    sweep converges: first tick adds 2.2.2.2, aging then evicts 1.1.1.1."""
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    handle = make_handle(session_id="sess_X")
    registry._handles["sess_X"] = handle
    dump = "\n".join(
        [
            "=filter=",
            "-P OUTPUT DROP",
            "-A OUTPUT -o lo -j ACCEPT",
            "-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT",
            "-A OUTPUT -p udp -m udp --dport 53 -j ACCEPT",
            "-A OUTPUT -p tcp -m tcp --dport 53 -j ACCEPT",
            "-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 80 -j ACCEPT",
            "-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 443 -j ACCEPT",
            # Extra-host-port proxy ACCEPT (non-80/443) must NOT be pinned.
            "-A OUTPUT -d 172.18.0.2/32 -p tcp -m tcp --dport 49152 -j ACCEPT",
            "=nat=",
            "-P OUTPUT ACCEPT",
            "-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 443 "
            "-j DNAT --to-destination 172.18.0.2:49152",
        ]
    )
    backend.sidecar_results = [
        _sidecar_result(stdout="api.github.com 2.2.2.2\n"),  # stamp-time resolve
        _sidecar_result(stdout=dump),  # live-rule read-back
    ]

    await registry._stamp_egress_state(
        handle,
        credential_hosts=frozenset({"api.github.com"}),
        limited_hosts=frozenset({"api.github.com"}),
        fallback_proxy_port=0,
        runtime=None,
    )

    state = registry._egress_states["sess_X"]
    assert state.pinned == {"api.github.com": {"1.1.1.1": 0}}
    assert (state.proxy_ip, state.proxy_port) == _PROXY


def test_unread_predicate_distinguishes_absent_from_authoritatively_empty() -> None:
    """The two layers must agree on ONE predicate, so it is pinned directly.

    Absent ⇒ unread (deletes refused, ``pinned`` held). Present-but-empty ⇒
    authoritative (deletes proceed, ``pinned`` advances). Conflating them is
    the whole bug class.
    """
    scope = {"credential_hosts": {"a.example"}, "limited_hosts": {"b.example"}}

    assert egress_unread_hosts(new_ips={"a.example": {"1.1.1.1"}}, **scope) == ["b.example"]
    assert egress_unread_hosts(new_ips={}, **scope) == ["a.example", "b.example"]
    # Present-with-empty-set is READ, not unread.
    assert egress_unread_hosts(new_ips={"a.example": set(), "b.example": set()}, **scope) == []


async def test_merge_holds_pinned_and_warns_when_inventory_omits_an_in_scope_host() -> None:
    """CALLER LAYER: a refusal must not be mistaken for a successful apply.

    The refusal script exits 0. If the caller advanced ``pinned`` on that
    success it would FORGET the IPs whose deletes were withheld — their rules
    stay installed, untracked and un-evictable forever. That inverts the
    discipline the nonzero-exit path already uses. So ``pinned`` is held and a
    warning is emitted, and the next tick retries the same delta: the refusal
    has an escalation path rather than being a terminal state.

    Reachability is stated honestly: this drives the real registry method but
    hand-seeds a ``pinned`` map missing an in-scope host, which today's stamp
    (``_seed_pinned_from_installed`` writes a key per host) does not produce.
    """
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    registry._handles["sess_X"] = make_handle(session_id="sess_X")
    # ``b.example`` is in scope but has NO key — the unread shape.
    state = _state(
        credential_hosts=frozenset({"a.example", "b.example"}),
        limited_hosts=frozenset(),
        pinned={"a.example": {"1.1.1.1": 0}},
    )
    registry._egress_states["sess_X"] = state
    before = {host: dict(ips) for host, ips in state.pinned.items()}

    with capture_logs() as logs:
        await registry._merge_egress_resolutions("sess_X", {"a.example": {"2.2.2.2"}})

    script = _sidecar_scripts(backend)[-1]
    # Adds still apply (an add only widens what is already permitted)...
    assert "-A OUTPUT -d 2.2.2.2" in script
    # ...but nothing is deleted, and the script exits 0 all the same.
    assert "-D OUTPUT" not in script
    # The caller must NOT bank that exit-0 as a completed delta.
    assert state.pinned == before
    assert any(entry.get("event") == "sandbox.egress_refresh_deletes_refused" for entry in logs)
    assert any(entry.get("unread_hosts") == ["b.example"] for entry in logs)
