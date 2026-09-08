"""Every external command the egress scripts run must come from the operator image.

Under ``runsc`` the egress rules cannot be installed from a netns-joining
sidecar — two runsc containers sharing a Linux netns get separate Sentries, so
the sidecar programs its own netstack and the target's tables stay empty
(#2310, gvisor#170). ``DockerBackend.run_netns_sidecar`` therefore ``docker
exec``s into the target container, which means the lockdown script runs inside
the TENANT's mount namespace, against a durable, tenant-writable root
filesystem.

That is only safe if *no* executable the script runs can be replaced by the
tenant. ``_runsc_operator_preamble`` is what buys that: it binds each command
name to a real file in the read-only operator image mount. The claim is easy to
break by accident — the security hole is a command the scripts use and the
preamble forgot, which resolves through ``PATH`` to whatever the tenant left
there. ``awk``/``sort``/``head`` (``resolve_ipv4``, and the ``$PROXY_IP``
lookup) were exactly that: a poisoned ``awk`` emitting ``0.0.0.0/0`` turns the
Limited allow-list into a blanket ACCEPT.

So this module executes the REAL generated scripts under bash with ``PATH``
pointing at a synthetic operator root containing ONLY the shadowed tools, and
records every command bash could not find. A miss means an unshadowed external
command — i.e. a tenant-controlled binary on the lockdown path.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from aios.sandbox import setup
from aios.sandbox.backends.docker import (
    _RUNSC_OPERATOR_COMMANDS,
    _RUNSC_OPERATOR_LIBRARY_PATH,
    _RUNSC_OPERATOR_LOADER,
    _RUNSC_OPERATOR_ROOT,
    _RUNSC_OPERATOR_SHELL,
    _runsc_operator_preamble,
)

# Stub bodies for the synthetic operator root. Real behaviour where the scripts
# depend on it (text tools), recording/permissive stubs where they don't.
_LOADER_STUB = (
    '#!/bin/sh\n# ld.so stub: drop "--library-path <dir>", exec the program\nshift 2\nexec "$@"\n'
)
# ``-S`` answers with a ruleset satisfying every read-back verify assertion, so
# a verify script's nonzero exit means the operator preamble broke it, not that
# the stub is unconvincing. ``-C``/``-D`` answer "absent" so the refresh
# script's append-if-missing / delete-if-present guards both take a branch.
_IPTABLES_SHOW = "\n".join(
    [
        "-P OUTPUT DROP",
        "-A OUTPUT -d 203.0.113.7/32 -p tcp -m tcp --dport 443 -j DNAT",
        *(f"-A OUTPUT -d {cidr} -j DROP" for cidr in setup._BROWSER_DENY_INTERNAL_CIDRS),
    ]
)
_IPTABLES_STUB = (
    "#!/bin/sh\n"
    'for a in "$@"; do case "$a" in\n'
    f"  -S) printf '%s\\n' '{_IPTABLES_SHOW}'; exit 0;;\n"
    "  -C) exit 1;;\n"
    "  -D) exit 1;;\n"
    "esac; done\n"
    "exit 0\n"
)
_GETENT_STUB = '#!/bin/sh\necho "203.0.113.7 STREAM $2"\nexit 0\n'


def _real(name: str) -> str:
    """Absolute path of a host tool, for stubs that must behave for real."""
    which = subprocess.run(["which", name], capture_output=True, text=True)
    path = which.stdout.strip()
    if not path:
        pytest.skip(f"host lacks {name}, cannot build a faithful operator root")
    return path


def _operator_root(tmp_path: Path) -> Path:
    """A synthetic ``_RUNSC_OPERATOR_ROOT`` holding exactly the shadowed tools."""
    root = tmp_path / "operator-root"

    def _write(rel: str, body: str) -> None:
        dest = root / rel.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write(_RUNSC_OPERATOR_LOADER, _LOADER_STUB)
    _write(_RUNSC_OPERATOR_SHELL, f'#!/bin/sh\nexec {_real("bash")} "$@"\n')
    (root / _RUNSC_OPERATOR_LIBRARY_PATH.lstrip("/")).mkdir(parents=True, exist_ok=True)
    for path in sorted(set(_RUNSC_OPERATOR_COMMANDS.values())):
        name = os.path.basename(path)
        if name in ("iptables-legacy", "ip6tables-legacy"):
            _write(path, _IPTABLES_STUB)
        elif name == "getent":
            _write(path, _GETENT_STUB)
        else:
            # grep/mawk/sort/head must behave: the scripts parse their output
            # and assert on their exit status.
            host = _real("awk" if name == "mawk" else name)
            _write(path, f'#!/bin/sh\nexec {host} "$@"\n')
    return root


def _run(script: str, tmp_path: Path) -> tuple[int, list[str], str]:
    """Run ``preamble + script`` against the synthetic root; return misses."""
    root = _operator_root(tmp_path)
    misses = tmp_path / "misses.log"
    # The preamble hard-codes the mount point; retarget it at the synthetic
    # root so the presence check and every ``operator_exec`` resolve here.
    preamble = _runsc_operator_preamble().replace(_RUNSC_OPERATOR_ROOT, str(root))
    # Any command that is neither a builtin nor one of the shadow functions
    # falls through to PATH; PATH holds only the operator root, so an
    # unshadowed external command lands here.
    trap = f'command_not_found_handle() {{ printf "%s\\n" "$1" >> {misses}; return 127; }}\n'
    # Keep the tenant's /etc/resolv.conf out of the unit-test host's way; the
    # substitution does not touch the command surface being asserted.
    script = script.replace("/etc/resolv.conf", str(tmp_path / "resolv.conf"))
    env = {
        "PATH": f"{root}/usr/sbin:{root}/usr/bin",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [_real("bash"), "-p", "-c", trap + preamble + script],
        env=env,
        capture_output=True,
        text=True,
    )
    found = misses.read_text().split() if misses.exists() else []
    return proc.returncode, found, proc.stderr


def _scripts() -> dict[str, str]:
    """Every script `aios.sandbox.setup` hands to ``run_netns_sidecar``."""
    dnat_target = ("aios-worker", 49152)
    return {
        "limited_lockdown_apply": setup._RESOLV_PREAMBLE
        + setup.build_iptables_script(
            {"example.com"},
            [("extra.example.com", 8080)],
            dnat_hosts=["api.secret.com"],
            dnat_target=dnat_target,
        ),
        "lockdown_verify": setup.build_lockdown_verify_script(["api.secret.com"]),
        "dnat_only_apply": setup._RESOLV_PREAMBLE
        + setup.build_secret_egress_dnat_script(["api.secret.com"], dnat_target),
        "dnat_only_verify": setup.build_lockdown_verify_script(
            ["api.secret.com"], assert_drop=False
        ),
        "egress_dump": setup.build_egress_dump_script(),
        "egress_resolve": setup.build_egress_resolve_script(["example.com"]),
        "egress_refresh": setup.build_egress_refresh_script(
            old_ips={"api.secret.com": {"198.51.100.1"}, "example.com": {"198.51.100.2"}},
            new_ips={"api.secret.com": {"203.0.113.1"}, "example.com": {"203.0.113.2"}},
            credential_hosts={"api.secret.com"},
            limited_hosts={"example.com"},
            dnat_target=("10.0.0.9", 49152),
        ),
        "browser_deny_internal": setup.build_browser_deny_internal_script(),
        "browser_deny_internal_verify": setup.build_browser_deny_internal_verify_script(),
    }


@pytest.mark.parametrize("name", sorted(_scripts()))
def test_script_runs_no_unshadowed_command(name: str, tmp_path: Path) -> None:
    """No egress script reaches a binary the operator preamble did not bind.

    A name reported here resolves through ``PATH`` at runtime — inside the
    tenant's mount namespace, that is a tenant-writable file with full control
    over the rules the lockdown installs.
    """
    rc, misses, stderr = _run(_scripts()[name], tmp_path)
    assert not misses, (
        f"{name} runs command(s) not bound to the operator image: {sorted(set(misses))}. "
        "Add them to _RUNSC_OPERATOR_COMMANDS (mapped to the REAL binary, never "
        "an /etc/alternatives symlink) or the runsc lockdown trusts tenant files."
    )
    assert rc == 0, f"{name} failed under the operator preamble: rc={rc} stderr={stderr}"


def test_preamble_fails_closed_when_operator_root_is_missing(tmp_path: Path) -> None:
    """An unmounted/incomplete operator root must abort, never fall back to PATH.

    Without the presence check a container created before the image mount
    existed would run the whole lockdown against tenant binaries and could still
    exit 0.
    """
    proc = subprocess.run(
        [_real("bash"), "-p", "-c", _runsc_operator_preamble() + "iptables -S OUTPUT"],
        env={"PATH": "/nonexistent", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 90, proc.stderr
    assert "operator tool root incomplete" in proc.stderr


def test_shadowed_paths_are_never_alternatives_symlinks() -> None:
    """``/etc/alternatives/*`` symlinks are ABSOLUTE and escape the mount.

    ``/usr/sbin/iptables``, ``/usr/sbin/ip6tables`` and ``/usr/bin/awk`` are all
    update-alternatives links in the sandbox image; resolving one inside the
    tenant's mount namespace lands on ``/etc/alternatives/<name>`` in the TENANT
    root. Only real files may sit behind a shadow function.
    """
    alternatives = {"/usr/sbin/iptables", "/usr/sbin/ip6tables", "/usr/bin/awk"}
    assert not alternatives & set(_RUNSC_OPERATOR_COMMANDS.values())


def test_shadow_targets_do_not_rely_on_usr_merge_symlinks() -> None:
    """``/bin``, ``/lib``, ``/lib64``, ``/sbin`` are symlinks — address the real path.

    Their targets are relative in debian today, so they happen to stay inside
    the mount; that is a base-image property, not a guarantee. Pin the canonical
    ``/usr``-rooted paths so a base-image bump cannot silently redirect the
    loader or the shell into the tenant root.
    """
    merged = ("/bin/", "/lib/", "/lib64/", "/sbin/")
    paths = [
        _RUNSC_OPERATOR_LOADER,
        _RUNSC_OPERATOR_LIBRARY_PATH + "/",
        _RUNSC_OPERATOR_SHELL,
        *_RUNSC_OPERATOR_COMMANDS.values(),
    ]
    assert [p for p in paths if p.startswith(merged)] == []
