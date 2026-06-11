"""TLS test helpers: mint server leaves off an :class:`EgressCA`.

Thin delegation to the production mint (``sandbox/egress_ca.mint_server_leaf``)
so tests exercise the exact leaf contract the egress proxy ships — DNS SAN,
``CA:FALSE``, EKU ``serverAuth``, and a keyid-only AuthorityKeyIdentifier whose
SKI matches any worker process's CA cert copy while the serial does not.
"""

from __future__ import annotations

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec

from aios.sandbox.egress_ca import EgressCA, mint_server_leaf


def mint_leaf(ca: EgressCA, hostname: str) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """A server leaf for ``hostname`` signed by ``ca`` (see ``mint_server_leaf``)."""
    return mint_server_leaf(ca, hostname)
