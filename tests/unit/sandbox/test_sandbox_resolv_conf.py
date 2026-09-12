"""The sandbox image must ship the embedded resolver in its own layer.

Under ``runsc`` the egress lockdown ``docker exec``s into the target sandbox
and chroots into the operator image mounted READ-ONLY (see
``DockerBackend.run_netns_sidecar``). Two consequences meet here:

* ``setup._RESOLV_PREAMBLE`` cannot write ``/etc/resolv.conf`` any more — the
  mount is read-only, and the ``|| true`` swallows the failure silently.
* ``getent`` therefore reads the resolver config *from the image layer*.

If that file is missing, glibc falls back to 127.0.0.1, nothing resolves, and
Limited networking yields an empty allow-list (total egress blackhole) while
Unrestricted silently skips the secret-egress DNAT. Neither failure is visible
without a live runsc daemon, and the runsc e2e suite runs on a weekly cron —
so the invariant is pinned here instead.

SCOPE: these are source-level pins — the baked file's content and the Dockerfile
instruction that bakes it. They cannot see the built image, so they cannot tell
a resolver that survived into the layer from one BuildKit stripped (aios#2410).
``tests/e2e/test_sandbox_image_contract.py::
test_image_layer_carries_the_embedded_dns_resolver`` is the only oracle for
that, and it needs a Docker daemon. Do not "fix" a red e2e by relaxing anything
here.
"""

from __future__ import annotations

import re
from pathlib import Path

from aios.sandbox import setup

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESOLV_CONF = _REPO_ROOT / "docker" / "sandbox-resolv.conf"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.sandbox"


def _nameservers(text: str) -> list[str]:
    return re.findall(r"(?m)^\s*nameserver\s+(\S+)\s*$", text)


def test_baked_nameserver_is_the_embedded_dns_address() -> None:
    """The baked file and the preamble must name the same resolver.

    They are two spellings of one fact: the runc sidecar writes the preamble's
    address into its own rootfs, the chrooted runsc exec reads the baked one.
    A drift between them makes Limited egress depend on which runtime provisioned
    the sandbox.
    """
    assert _nameservers(_RESOLV_CONF.read_text()) == [setup._EMBEDDED_DNS_ADDRESS], (
        f"{_RESOLV_CONF.name} must declare exactly one nameserver and it must be "
        f"setup._EMBEDDED_DNS_ADDRESS ({setup._EMBEDDED_DNS_ADDRESS!r})"
    )
    assert f"nameserver {setup._EMBEDDED_DNS_ADDRESS}" in setup._RESOLV_PREAMBLE


def test_dockerfile_copies_the_resolver_after_every_run_step() -> None:
    """COPY (not RUN), ``--link``, and last.

    Docker mounts its own ``/etc/resolv.conf`` over the build sandbox, so a
    ``RUN`` redirect writes to the mount and never reaches the layer — COPY is
    the only way to bake one. And the COPY has to come after the ``apt-get``
    steps: those need the daemon's resolver, not the netns-local one.

    ``--link`` is pinned because a plain COPY in exactly this position was
    observed to land a 0-byte file in the built image (aios#2410); the linked
    layer is built over scratch and merged on top rather than written through
    the parent snapshot that carries BuildKit's mount placeholder. Pinning the
    flag keeps a well-meaning "simplify" from silently reintroducing the empty
    resolver — but it pins the INSTRUCTION, not the resulting layer. See the
    module docstring.
    """
    lines = _DOCKERFILE.read_text().splitlines()
    # Match on the DESTINATION, not on the flags: a plain ``COPY`` to this path
    # is the exact regression being pinned, so it has to be found and then
    # rejected by name — not silently missed by a flag-sensitive pattern.
    copies = [
        i
        for i, line in enumerate(lines)
        if re.match(r"^COPY (?:--\S+ )*\S+ /etc/resolv\.conf\s*$", line)
    ]
    assert len(copies) == 1, "expected exactly one COPY of /etc/resolv.conf in Dockerfile.sandbox"
    assert lines[copies[0]] == "COPY --link docker/sandbox-resolv.conf /etc/resolv.conf", (
        "the resolver COPY must keep ``--link``; a plain COPY to /etc/resolv.conf "
        "lands a 0-byte file in the built image (aios#2410)"
    )

    runs = [i for i, line in enumerate(lines) if line.startswith("RUN ")]
    assert runs and copies[0] > max(runs), (
        "the /etc/resolv.conf COPY precedes a RUN step — that RUN would try to "
        "resolve through the netns-local embedded DNS, which does not exist at "
        "build time"
    )
