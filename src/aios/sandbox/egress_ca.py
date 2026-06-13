"""Deterministic egress-proxy CA, derived from ``AIOS_EGRESS_CA_KEY``.

The secret-egress proxy (the ``git_proxy.py`` generalization that swaps
credential placeholders for real values at the worker boundary) must
terminate TLS for arbitrary allowlisted hosts, which means every sandbox
has to trust an aios-controlled CA. This module is that CA.

**Zero stored state.** The CA private key is an HKDF subkey of the required
``AIOS_EGRESS_CA_KEY`` secret (domain context :data:`EGRESS_CA_HKDF_INFO`),
so every worker sharing that key derives the *same* keypair — no table, no
bootstrap race, no key file. The self-signed cert is regenerated per process:
its serial and validity window vary, but the subject DN and keypair are
deterministic, and trust anchors match on subject + public key, so a leaf
minted by one worker process chains against the cert copy any other process
installed.

Consequences operators should know:

* **Blast radius**: ``AIOS_VAULT_KEY`` holders can decrypt at-rest rows but
  cannot mint sandbox-trusted TLS certificates. The egress CA has its own
  separately-rotated secret and runbook.
* **The CA is unconstrained** (no x509 name constraints — allowlists are
  per-credential and dynamic, a static constraint can't express them).
  Host scoping is enforced *solely* at leaf-mint time in the egress
  proxy; that check is security-critical and must fail closed.

Leaf contract for the egress proxy (#876): leaves MUST carry a DNS SAN
(CN-only verification is deprecated fallback behavior) and a
**keyid-only** AuthorityKeyIdentifier — the SKI is a hash of the
deterministic public key, so it matches across processes, while the
serial does not and must never appear in chain references.
"""

from __future__ import annotations

import functools
from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from aios.config import get_settings
from aios.crypto.vault import KEY_BYTES, CryptoBox

# HKDF domain-separation context for the CA seed; bump the suffix if the
# derivation scheme ever changes incompatibly.
EGRESS_CA_HKDF_INFO = "aios-egress-ca-v1"

# Server-leaf validity window. Short by design — leaves are minted per SNI
# host for one session's lifetime, never persisted.
LEAF_VALIDITY_DAYS = 7

# Debian convention: drop-in .crt files under this directory are folded
# into the aggregate bundle by ``update-ca-certificates``.
CA_CERT_SANDBOX_PATH = "/usr/local/share/ca-certificates/aios-egress-ca.crt"
SYSTEM_CA_BUNDLE_PATH = "/etc/ssl/certs/ca-certificates.crt"

# Injected into every sandbox so OpenSSL-based clients (curl, python ssl,
# requests/httpx via REQUESTS_CA_BUNDLE) and Node all see the aios CA.
# The bundle paths follow the default image's Debian layout and exist
# even before the CA install step runs; NODE_EXTRA_CA_CERTS is additive
# and points at the drop-in file itself. Merged early in the sandbox env
# (see ``spec._assemble_plan``) so an environment's ``env`` config can
# override them — the escape hatch for custom images (#724) with a
# non-Debian trust-store layout.
TRUST_STORE_ENV: dict[str, str] = {
    "SSL_CERT_FILE": SYSTEM_CA_BUNDLE_PATH,
    "REQUESTS_CA_BUNDLE": SYSTEM_CA_BUNDLE_PATH,
    "NODE_EXTRA_CA_CERTS": CA_CERT_SANDBOX_PATH,
}

_CA_SUBJECT = x509.Name(
    [
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "aios"),
        x509.NameAttribute(NameOID.COMMON_NAME, "aios Egress CA"),
    ]
)


class EgressCA:
    """A CA keypair + self-signed cert deterministically built from ``seed``.

    The private key is the seed reduced into the P-256 scalar range; the
    cert is freshly generated per instance (random serial, validity
    anchored at construction time) over the deterministic subject and
    key, which is all chain verification matches on.
    """

    def __init__(self, seed: bytes) -> None:
        if len(seed) != 32:
            raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
        n = ec.SECP256R1.group_order
        self.private_key = ec.derive_private_key(
            int.from_bytes(seed, "big") % (n - 1) + 1, ec.SECP256R1()
        )
        now = datetime.now(UTC)
        self.certificate = (
            x509.CertificateBuilder()
            .subject_name(_CA_SUBJECT)
            .issuer_name(_CA_SUBJECT)
            .public_key(self.private_key.public_key())
            .serial_number(x509.random_serial_number())
            # Backdate an hour so a sandbox whose clock trails the worker
            # slightly still accepts a just-generated cert.
            .not_valid_before(now - timedelta(hours=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(self.private_key.public_key()),
                critical=False,
            )
            .sign(self.private_key, hashes.SHA256())
        )
        self.cert_pem = self.certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


@functools.cache
def _egress_ca_from_seed(seed: bytes) -> EgressCA:
    return EgressCA(seed)


def derive_egress_ca_seed() -> bytes:
    """The 32-byte CA seed: an HKDF subkey of ``AIOS_EGRESS_CA_KEY``.

    The single authoritative derivation — :func:`get_egress_ca` and tests
    that mint a leaf from a *fresh* :class:`EgressCA` (the cross-process
    contract) both call it, so the scheme can never silently fork.
    """
    key = get_settings().egress_ca_key.get_secret_value()
    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=b"aios-vault-hkdf-v1",
        info=EGRESS_CA_HKDF_INFO.encode(),
    ).derive(CryptoBox.from_base64(key, env_name="AIOS_EGRESS_CA_KEY").key_bytes)


def get_egress_ca() -> EgressCA:
    """The worker's egress CA, derived from ``AIOS_EGRESS_CA_KEY``.

    Cached per seed (not per process), so tests or a rare CA rotation path
    that swaps the env secret get a matching CA without process-global state.
    """
    return _egress_ca_from_seed(derive_egress_ca_seed())


def mint_server_leaf(
    ca: EgressCA, hostname: str
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """A server leaf for ``hostname`` signed by ``ca`` — the leaf contract
    pinned in this module's docstring.

    Carries a DNS SAN (the authoritative identity; CN-only verification is
    deprecated), ``CA:FALSE``, EKU ``serverAuth``, a short validity window,
    and a **keyid-only** AuthorityKeyIdentifier (``from_issuer_public_key``).
    The SKI hashes the deterministic public key, so it matches any worker
    process's CA cert copy; the CA serial does not and must never appear in
    a chain reference. This is the single authoritative mint — the egress
    proxy and the ``tests/helpers/tls.py`` helper both call it.

    The subject is left EMPTY and the SAN is marked critical. The hostname is
    deliberately NOT placed in the CN: X.509 caps the CN at 64 chars
    (ub-common-name), but a credential's ``allowed_hosts`` admits any DNS name
    up to 253 chars — a 65-253-char host would make a CN-bearing mint raise
    and render that allowed host permanently unreachable through the proxy.
    The DNS SAN has no such limit, and an empty subject is valid precisely
    when the SAN is critical (RFC 5280 §4.2.1.6).
    """
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([]))
        .issuer_name(ca.certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Backdate an hour so a sandbox whose clock trails the worker still
        # accepts a just-minted leaf (mirrors the CA cert's backdating).
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=LEAF_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=True)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.private_key.public_key()),
            critical=False,
        )
        .sign(ca.private_key, hashes.SHA256())
    )
    return cert, key


__all__ = [
    "CA_CERT_SANDBOX_PATH",
    "EGRESS_CA_HKDF_INFO",
    "LEAF_VALIDITY_DAYS",
    "SYSTEM_CA_BUNDLE_PATH",
    "TRUST_STORE_ENV",
    "EgressCA",
    "derive_egress_ca_seed",
    "get_egress_ca",
    "mint_server_leaf",
]
