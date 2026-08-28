"""Structural pins for ``docker/seccomp-browser.json`` (jarbot#106 Phase 2).

The authored browser profile has exactly one job: re-permit the unprivileged
USER/PID/NET namespaces Chromium's own sandbox needs while keeping MOUNT (and
CGROUP/UTS/IPC) namespace creation denied. These tests pin the JSON shape so a
re-vendor or an edit cannot silently regress the mask — the failure mode the
design review caught in draft (a mask omitting CLONE_NEWNS would have
re-enabled mount-namespace creation). The live-kernel counterpart runs in
``tests/e2e/test_browser_image_contract.py`` against a real container.

Pure JSON inspection — no Docker required, runs in the normal unit lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"

# linux/sched.h CLONE_NEW* flag values.
_NEWNS = 0x20000
_NEWCGROUP = 0x2000000
_NEWUTS = 0x4000000
_NEWIPC = 0x8000000

# The namespaces the browser profile keeps DENYING. Everything Chromium's
# zygote needs (NEWUSER | NEWPID | NEWNET) is deliberately absent from the
# mask; NEWNS is the must-not-regress bit.
_DENIED_MASK = _NEWNS | _NEWCGROUP | _NEWUTS | _NEWIPC  # 0x0E020000 = 235012096


@pytest.fixture(scope="module")
def profile() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_DOCKER_DIR / "seccomp-browser.json").read_text())
    return data


def test_default_action_is_eperm(profile: dict[str, Any]) -> None:
    """Anything unmatched is denied with EPERM (never KILL)."""
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert profile["defaultErrnoRet"] == 1


def test_authored_block_is_first_and_masks_exactly_the_denied_namespaces(
    profile: dict[str, Any],
) -> None:
    """The authored clone/unshare allow sits at index 0 (ahead of every
    vendored rule) and its mask is EXACTLY the four denied namespaces."""
    block = profile["syscalls"][0]
    assert sorted(block["names"]) == ["clone", "unshare"]
    assert block["action"] == "SCMP_ACT_ALLOW"
    (arg,) = block["args"]
    assert arg == {
        "index": 0,
        "value": _DENIED_MASK,
        "valueTwo": 0,
        "op": "SCMP_CMP_MASKED_EQ",
    }
    # s390* pass clone flags in arg1; excluded rather than silently misread.
    assert block["excludes"] == {"arches": ["s390", "s390x"]}


def test_no_other_unconditional_clone_or_unshare_allow(profile: dict[str, Any]) -> None:
    """Nothing after the authored block may allow clone/unshare more broadly:
    every later allow must be arg-filtered or capability-gated, so the
    authored mask stays the effective policy for plain processes."""
    for block in profile["syscalls"][1:]:
        if block.get("action") != "SCMP_ACT_ALLOW":
            continue
        if not set(block.get("names", [])) & {"clone", "unshare"}:
            continue
        gated = bool(block.get("args")) or bool((block.get("includes") or {}).get("caps"))
        assert gated, f"ungated clone/unshare allow after the authored block: {block}"


def test_clone3_stays_enosys(profile: dict[str, Any]) -> None:
    """clone3 must return ENOSYS so glibc falls back to clone, where the
    authored arg0 filter applies (clone3 passes flags in a struct seccomp
    cannot inspect — allowing it would bypass the mask entirely)."""
    rules = [b for b in profile["syscalls"] if "clone3" in b.get("names", [])]
    enosys = [b for b in rules if b.get("action") == "SCMP_ACT_ERRNO" and b.get("errnoRet") == 38]
    assert enosys, "no clone3 → ENOSYS rule; glibc would not fall back to filtered clone"
    for rule in rules:
        if rule.get("action") == "SCMP_ACT_ALLOW":
            caps = (rule.get("includes") or {}).get("caps")
            assert caps, f"ungated clone3 allow would bypass the arg0 mask: {rule}"


def test_ptrace_stays_allowed_for_crashpad(profile: dict[str, Any]) -> None:
    """Unlike seccomp-sandbox.json there is deliberately NO flat ptrace deny:
    Chromium's crashpad handler ptraces its own processes, and no agent code
    runs in this container. The vendored kernel>=4.8 allow must survive."""
    for block in profile["syscalls"]:
        names = set(block.get("names", []))
        if "ptrace" in names and block.get("action") == "SCMP_ACT_ERRNO":
            pytest.fail(f"ptrace deny block present — crashpad would break: {sorted(names)}")
    allowed = any(
        "ptrace" in block.get("names", []) and block.get("action") == "SCMP_ACT_ALLOW"
        for block in profile["syscalls"]
    )
    assert allowed, "no ptrace allow rule found in the vendored base"


def test_vendored_base_identical_to_sandbox_profile(profile: dict[str, Any]) -> None:
    """Both profiles claim the same verbatim-vendored moby v24.0.7 base
    (browser = 1 authored block + base; sandbox = 2 authored blocks + base).
    A re-vendor of ONE file would silently fork that shared base — this pin
    turns it into a failure that names the maintenance step: re-vendor both,
    re-applying each file's authored front blocks."""
    sandbox = json.loads((_DOCKER_DIR / "seccomp-sandbox.json").read_text())
    assert profile["syscalls"][1:] == sandbox["syscalls"][2:], (
        "the vendored syscall blocks of seccomp-browser.json and seccomp-sandbox.json have diverged"
    )
    for key in ("archMap", "defaultAction", "defaultErrnoRet"):
        assert profile[key] == sandbox[key], f"top-level {key!r} diverged between the profiles"
