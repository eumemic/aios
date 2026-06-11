"""TLS test helpers: mint server leaves off an :class:`EgressCA`.

Mints leaves the way the secret-egress proxy must (the contract pinned
in ``sandbox/egress_ca.py``): DNS SAN (CN-only verification is
deprecated fallback behavior) and a keyid-only AuthorityKeyIdentifier —
the SKI hashes the deterministic public key, so it matches any worker
process's CA cert copy, while the serial does not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID

from aios.sandbox.egress_ca import EgressCA


def mint_leaf(ca: EgressCA, hostname: str) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """A server leaf for ``hostname`` signed by ``ca``."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, hostname)]))
        .issuer_name(ca.certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.private_key.public_key()),
            critical=False,
        )
        .sign(ca.private_key, hashes.SHA256())
    )
    return cert, key
