"""runsc OCI seccomp + empty-/tmp tmpfs: the two gVisor-validation RED legs.

(a) gVisor ignores seccomp ``errnoRet``, so clone3 ENOSYS becomes EPERM and
    python/node cannot start threads. The runsc create path prepends clone3
    ALLOW; unshare CLONE_NEWUSER stays denied.
(b) gVisor mounts tmpfs over empty ``/tmp``, hiding writes from docker
    commit. The sandbox image plants a sentinel so ``/tmp`` stays on the
    rootfs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import (
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    Mount,
    SandboxSpec,
)
from aios.sandbox.backends.docker import _runsc_seccomp_profile, _seccomp_opt

_REPO_ROOT = Path(__file__).parents[3]
_PROFILE = _REPO_ROOT / "docker" / "seccomp-sandbox.json"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.sandbox"


def _spec(*, runtime: str | None, seccomp_profile: str) -> SandboxSpec:
    return SandboxSpec(
        session_id="sess_seccomp",
        instance_id="inst_seccomp",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={
            MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
            INSTANCE_LABEL_KEY: "inst_seccomp",
            SESSION_LABEL_KEY: "sess_seccomp",
        },
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image="ghcr.io/eumemic/aios-sandbox:latest",
        runtime=runtime,
        seccomp_profile=seccomp_profile,
    )


@pytest.fixture(autouse=True)
def _clear_profile_cache() -> None:
    _runsc_seccomp_profile.cache_clear()


def test_runsc_seccomp_profile_allows_clone3_before_errno() -> None:
    path = Path(_runsc_seccomp_profile(str(_PROFILE)))
    profile = json.loads(path.read_text())
    syscalls = profile["syscalls"]
    clone3_idx = [i for i, blk in enumerate(syscalls) if "clone3" in blk.get("names", [])]
    assert clone3_idx, "clone3 vanished from the runsc profile"
    first = syscalls[clone3_idx[0]]
    assert first["action"] == "SCMP_ACT_ALLOW"
    assert not first.get("args")
    assert not first.get("includes")
    assert clone3_idx[0] == 0, "clone3 ALLOW must be first-match against later ENOSYS/EPERM"


def test_runsc_seccomp_profile_still_denies_unshare() -> None:
    path = Path(_runsc_seccomp_profile(str(_PROFILE)))
    profile = json.loads(path.read_text())
    denied = [
        blk
        for blk in profile["syscalls"]
        if blk.get("action") == "SCMP_ACT_ERRNO" and "unshare" in blk.get("names", [])
    ]
    assert denied, "unshare EPERM (CLONE_NEWUSER) must survive the runsc clone3 ALLOW"
    clone_flat_deny = [
        blk
        for blk in profile["syscalls"]
        if blk.get("action") == "SCMP_ACT_ERRNO"
        and "clone" in blk.get("names", [])
        and not blk.get("args")
    ]
    assert not clone_flat_deny, "a flat clone deny would brick pthreads"


def test_seccomp_opt_is_runsc_only() -> None:
    authored = str(_PROFILE)
    assert _seccomp_opt(_spec(runtime=None, seccomp_profile=authored)) == authored
    assert _seccomp_opt(_spec(runtime="runsc", seccomp_profile="unconfined")) == "unconfined"
    derived = _seccomp_opt(_spec(runtime="runsc", seccomp_profile=authored))
    assert derived != authored
    assert Path(derived).is_file()


def test_runsc_profile_is_the_authored_one_plus_exactly_the_clone3_allow() -> None:
    """The derived copy must widen the authored policy by ONE rule and no more.

    The clone3 ALLOW is a deliberate, documented hole (gVisor implements clone3
    natively, and its flags live in a struct seccomp cannot filter, so
    ``clone3(CLONE_NEWUSER)`` is reachable under runsc). Pinning the rest of the
    profile byte-for-byte is what keeps that hole from quietly growing a second
    one the next time this derivation is touched.
    """
    authored = json.loads(_PROFILE.read_text())
    derived = json.loads(Path(_runsc_seccomp_profile(str(_PROFILE))).read_text())
    inserted = derived["syscalls"][0]
    assert inserted["names"] == ["clone3"] and inserted["action"] == "SCMP_ACT_ALLOW"
    assert derived["syscalls"][1:] == authored["syscalls"]
    assert {k: v for k, v in derived.items() if k != "syscalls"} == {
        k: v for k, v in authored.items() if k != "syscalls"
    }


def test_runsc_profile_keeps_the_unconditional_namespace_deny() -> None:
    """What still holds after the clone3 ALLOW: the #807 deny block is
    unconditional and ahead of every base ALLOW, so a tenant who does reach a
    user namespace via clone3 still cannot mount/setns/unshare inside it."""
    derived = json.loads(Path(_runsc_seccomp_profile(str(_PROFILE))).read_text())
    syscalls = derived["syscalls"]
    deny_idx = next(
        i
        for i, blk in enumerate(syscalls)
        if blk.get("action") == "SCMP_ACT_ERRNO" and {"mount", "setns"} <= set(blk.get("names", []))
    )
    deny = syscalls[deny_idx]
    assert not deny.get("args") and not deny.get("includes"), "the deny must be unconditional"
    later_allows = [
        i
        for i, blk in enumerate(syscalls)
        if i > deny_idx
        and blk.get("action") == "SCMP_ACT_ALLOW"
        and set(blk.get("names", [])) & set(deny["names"])
    ]
    assert all(syscalls[i].get("args") or syscalls[i].get("includes") for i in later_allows), (
        "an unconditional ALLOW after the deny would shadow it"
    )


def test_sandbox_image_keeps_tmp_nonempty_for_gvisor() -> None:
    """runsc `mountTmp` overlays tmpfs on empty /tmp; a sentinel prevents that."""
    text = _DOCKERFILE.read_text()
    assert "touch /tmp/.aios-keep" in text
    assert "mountTmp" in text
