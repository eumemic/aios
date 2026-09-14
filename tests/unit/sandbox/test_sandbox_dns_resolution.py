"""The egress scripts must resolve through an explicitly-named DNS server.

Every hostname in a Limited allow-list, every credential host behind the
secret-egress DNAT and the proxy alias itself are turned into iptables rules by
one emitted shell function, ``setup._RESOLVE_IPV4_FN``. Whatever answers it
decides which addresses get an ACCEPT (or a DNAT) rule, so *which resolver it
asks* is a security property, not a detail.

It used to ask glibc (``getent ahostsv4``), and glibc reads the servers out of
``/etc/resolv.conf`` and nothing else. Neither sidecar shape has a usable one:

* runc — Docker writes no ``resolv.conf`` for a ``--network container:<id>``
  sidecar, so it inherits the IMAGE's file;
* runsc — the exec chroots into the operator image mounted READ-ONLY, so it
  reads that image's file and nothing can write another one.

And the file cannot be baked either: BuildKit treats ``/etc/resolv.conf`` as
runtime-managed and commits an EMPTY entry for any ``COPY`` to that path —
observed on a CI-built image with a plain ``COPY`` and again with ``COPY
--link`` (aios#2410, moby/buildkit#1267; DONE.md carries the evidence chain).

So the resolver is passed as an argument instead: ``busybox nslookup <host>
127.0.0.11``. These are the source-level pins for that contract. The end-to-end
oracle — that the shipped image's busybox really answers this way against
Docker's embedded DNS — is
``tests/e2e/test_sandbox_image_contract.py``, which needs a daemon.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from aios.sandbox import setup
from aios.sandbox.backends.docker import _RUNSC_OPERATOR_STATIC_COMMANDS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.sandbox"


def _scripts() -> dict[str, str]:
    """Every script `aios.sandbox.setup` hands to ``run_netns_sidecar``."""
    dnat_target = ("aios-worker", 49152)
    return {
        "limited_lockdown_apply": setup.build_iptables_script(
            {"example.com"},
            [("extra.example.com", 8080)],
            dnat_hosts=["api.secret.com"],
            dnat_target=dnat_target,
        ),
        "dnat_only_apply": setup.build_secret_egress_dnat_script(["api.secret.com"], dnat_target),
        "egress_resolve": setup.build_egress_resolve_script(["example.com"]),
    }


def test_resolver_is_named_explicitly() -> None:
    """The emitted helper passes the embedded resolver's address as an argument."""
    assert f'busybox nslookup "$1" {setup._EMBEDDED_DNS_ADDRESS}' in setup._RESOLVE_IPV4_FN


@pytest.mark.parametrize("name", sorted(_scripts()))
def test_no_script_touches_resolv_conf_or_getent(name: str) -> None:
    """Nothing on the lockdown path may depend on a resolver config FILE.

    A reintroduced ``getent`` silently resolves through whichever
    ``/etc/resolv.conf`` the sidecar happened to inherit; a reintroduced
    ``printf ... > /etc/resolv.conf`` preamble silently writes nothing under
    runsc (read-only operator mount) and is swallowed by its ``|| true``. Both
    failure modes are invisible until Limited egress blackholes in production.
    """
    script = _scripts()[name]
    assert "/etc/resolv.conf" not in script
    assert "getent" not in script


def test_busybox_is_shadowed_as_a_static_operator_command() -> None:
    """The runsc preamble must bind ``busybox`` to the operator image.

    ``resolve_ipv4`` runs inside the TENANT's mount namespace under runsc, so an
    unbound ``busybox`` resolves through ``PATH`` to a tenant-writable file that
    gets to choose every address the allow-list ends up containing. It belongs
    in the STATIC family: busybox carries no ``PT_INTERP``, so running it
    through the operator image's ``ld.so`` (as every dynamic command is) would
    fail outright.
    """
    assert "busybox" in _RUNSC_OPERATOR_STATIC_COMMANDS


def test_dockerfile_bakes_no_resolver() -> None:
    """No ``COPY`` to ``/etc/resolv.conf`` — BuildKit commits it empty.

    This is the regression guard for the fix: a well-meaning "the chroot needs a
    nameserver, just COPY one in" lands a 0-byte file in the layer, which reads
    as a resolver that exists and works right up until nothing resolves.
    """
    dockerfile = _DOCKERFILE.read_text()
    assert "/etc/resolv.conf" not in dockerfile, (
        "docker/Dockerfile.sandbox references /etc/resolv.conf again — BuildKit "
        "commits an EMPTY entry for any COPY to that path (aios#2410); the "
        "resolver is named explicitly in setup._RESOLVE_IPV4_FN instead"
    )


# The real busybox 1.35 ``nslookup`` shape: the resolver's own address first,
# then the answers. ``$2`` is the queried name, ``$3`` the server.
_BUSYBOX_STUB = (
    "#!/bin/sh\n"
    '[ "$1" = nslookup ] || { echo "busybox: unknown applet $1" >&2; exit 1; }\n'
    'case "$2" in\n'
    "  missing.example.com)\n"
    '    printf \'Server:\\t\\t%s\\nAddress:\\t%s:53\\n\\n\' "$3" "$3"\n'
    "    echo \"nslookup: can't resolve '$2': Name or service not known\" >&2\n"
    "    exit 1;;\n"
    "esac\n"
    "printf 'Server:\\t\\t%s\\nAddress:\\t%s:53\\n\\nName:\\t%s\\n"
    'Address: 127.0.0.11\\nAddress: 203.0.113.7\\nAddress: 2001:db8::1\\nAddress: 198.51.100.9\\n\' "$3" "$3" "$2"\n'
)


def _resolve(host: str, tmp_path: Path) -> list[str]:
    """Run the REAL emitted helper against a stub busybox; return its output."""
    stub = tmp_path / "busybox"
    stub.write_text(_BUSYBOX_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    proc = subprocess.run(
        ["bash", "-c", f"{setup._RESOLVE_IPV4_FN}\nresolve_ipv4 {host}"],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    return proc.stdout.split()


def test_only_ipv4_answers_survive_the_parse(tmp_path: Path) -> None:
    """A answers only — never the AAAA, never the resolver's own address.

    The AAAA would be handed to an IPv4-only ``iptables -d``, which errors and,
    under ``set -e``, aborts the whole apply — a Limited sandbox that never
    finishes provisioning the moment an allowed host gains a v6 record. The
    server's address is worse: silently ACCEPTing 127.0.0.11, or picking it up
    as ``$PROXY_IP`` and DNATing every credential host at the resolver.
    """
    assert _resolve("example.com", tmp_path) == ["198.51.100.9", "203.0.113.7"]


def test_a_resolution_miss_yields_nothing(tmp_path: Path) -> None:
    """Fail closed: an unresolvable host gets no addresses, so it gets no rule."""
    assert _resolve("missing.example.com", tmp_path) == []
