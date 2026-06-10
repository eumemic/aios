"""Sanity checks on the authored seccomp profile (issue #807).

The profile at ``docker/seccomp-sandbox.json`` is a verbatim vendored copy
of moby's pinned ``default.json`` (v24.0.7) with two authored blocks
PREPENDED to the front of the ``syscalls`` array:

1. an arg-filtered ALLOW for ``unshare`` with no CLONE_NEW* bit set
   (mask ``0x7E020000`` == ``2114060288``), and
2. a flat ERRNO (EPERM) deny block for the namespace/key/trace syscalls
   the issue calls out — deliberately EXCLUDING ``clone``/``clone3`` so the
   base's arg-filtered ``clone`` allow (pthreads/Node) survives.

seccomp is first-match-wins, so prepending the deny block makes the
deny-list hold even if the vendored base drifts or a cap is added. These
tests pin that structure; they do not exercise the kernel (that's the e2e
suite).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# From tests/unit/sandbox/test_X.py, the repo root is parents[3]
# (test_seccomp_profile.py -> sandbox -> unit -> tests -> repo root).
REPO_ROOT = Path(__file__).parents[3]
PROFILE_PATH = REPO_ROOT / "docker" / "seccomp-sandbox.json"

# OR of all seven CLONE_NEW* bits: NEWNS 0x20000, NEWCGROUP 0x2000000,
# NEWUTS 0x4000000, NEWIPC 0x8000000, NEWUSER 0x10000000, NEWPID 0x20000000,
# NEWNET 0x40000000 == 0x7E020000.
NAMESPACE_MASK = 2114060288

# Names the issue requires explicitly denied (EPERM, not KILL).
DENY_NAMES = {
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "mount",
    "umount",
    "umount2",
    "keyctl",
    "add_key",
    "request_key",
    "bpf",
    "userfaultfd",
    "setns",
    "unshare",
}


@pytest.fixture(scope="module")
def profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text())


def test_profile_parses(profile: dict[str, object]) -> None:
    """The file is valid JSON with the moby top-level shape and a deny default."""
    assert {"defaultAction", "archMap", "syscalls"} <= set(profile)
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"


def test_deny_entries_present_with_errno(profile: dict[str, object]) -> None:
    """Every required name appears in at least one SCMP_ACT_ERRNO block."""
    syscalls = profile["syscalls"]
    assert isinstance(syscalls, list)
    denied: set[str] = set()
    for blk in syscalls:
        if blk.get("action") == "SCMP_ACT_ERRNO":
            denied.update(blk.get("names", []))
    missing = DENY_NAMES - denied
    assert not missing, f"required deny syscalls not in any ERRNO block: {sorted(missing)}"


def test_no_flat_clone_deny(profile: dict[str, object]) -> None:
    """No flat (argless) ERRNO deny for ``clone`` — that would brick pthreads.

    ``clone`` must still carry at least one ALLOW block with args (the base's
    namespace-masked arg filter).
    """
    syscalls = profile["syscalls"]
    assert isinstance(syscalls, list)
    for blk in syscalls:
        if blk.get("action") == "SCMP_ACT_ERRNO" and "clone" in blk.get("names", []):
            assert blk.get("args"), (
                "found a flat (argless) ERRNO deny for clone — would break threads"
            )
    clone_allow_with_args = [
        blk
        for blk in syscalls
        if blk.get("action") == "SCMP_ACT_ALLOW"
        and "clone" in blk.get("names", [])
        and blk.get("args")
    ]
    assert clone_allow_with_args, "clone lost its arg-filtered ALLOW block"


def test_clone_argfilter_uses_namespace_mask(profile: dict[str, object]) -> None:
    """Some ``clone`` ALLOW block masks arg0 against the CLONE_NEW* bits."""
    syscalls = profile["syscalls"]
    assert isinstance(syscalls, list)
    found = False
    for blk in syscalls:
        if blk.get("action") == "SCMP_ACT_ALLOW" and "clone" in blk.get("names", []):
            for arg in blk.get("args", []):
                if arg.get("op") == "SCMP_CMP_MASKED_EQ" and arg.get("value") == NAMESPACE_MASK:
                    found = True
    assert found, f"no clone ALLOW block masks against {NAMESPACE_MASK}"


def test_unshare_carries_arg_conditions(profile: dict[str, object]) -> None:
    """``unshare`` has a masked-eq ALLOW that PRECEDES the flat ERRNO deny.

    seccomp is first-match-wins: the benign ``unshare(0)`` allow must come
    before the catch-all deny so namespace-creating unshare is denied while
    no-flag unshare is permitted.
    """
    syscalls = profile["syscalls"]
    assert isinstance(syscalls, list)
    allow_idx: int | None = None
    deny_idx: int | None = None
    for i, blk in enumerate(syscalls):
        names = blk.get("names", [])
        if "unshare" not in names:
            continue
        if blk.get("action") == "SCMP_ACT_ALLOW":
            masked = [
                arg
                for arg in blk.get("args", [])
                if arg.get("op") == "SCMP_CMP_MASKED_EQ" and arg.get("value") == NAMESPACE_MASK
            ]
            if masked and allow_idx is None:
                allow_idx = i
        elif blk.get("action") == "SCMP_ACT_ERRNO":
            if deny_idx is None:
                deny_idx = i
    assert allow_idx is not None, "no masked-eq ALLOW block for unshare"
    assert deny_idx is not None, "no ERRNO deny block for unshare"
    assert allow_idx < deny_idx, (
        f"unshare ALLOW (idx {allow_idx}) must precede the flat deny (idx {deny_idx})"
    )


def test_clone3_not_in_authored_flat_deny(profile: dict[str, object]) -> None:
    """The authored flat-deny block (the one carrying setns/bpf) excludes clone3.

    A flat ERRNO on clone3 would break glibc fork/thread paths that probe
    clone3 first; the base already ERRNOs clone3 for non-CAP_SYS_ADMIN, which
    is enough.
    """
    syscalls = profile["syscalls"]
    assert isinstance(syscalls, list)
    authored = [
        blk
        for blk in syscalls
        if blk.get("action") == "SCMP_ACT_ERRNO"
        and "setns" in blk.get("names", [])
        and "bpf" in blk.get("names", [])
    ]
    assert authored, "authored flat-deny block (setns+bpf) not found"
    for blk in authored:
        assert "clone3" not in blk.get("names", []), "clone3 must not be in the authored flat deny"
        assert "clone" not in blk.get("names", []), "clone must not be in the authored flat deny"


def test_settings_default_path_points_at_repo_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Settings().sandbox_seccomp_profile`` defaults to the vendored file."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_SANDBOX_SECCOMP_PROFILE", raising=False)

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.sandbox_seccomp_profile.endswith("docker/seccomp-sandbox.json")
    assert Path(s.sandbox_seccomp_profile).exists()
