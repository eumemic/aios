"""Deterministic egress-proxy CA, derived from the vault master key.

The secret-egress proxy (the ``git_proxy.py`` generalization that swaps
credential placeholders for real values at the worker boundary) must
terminate TLS for arbitrary allowlisted hosts, which means every sandbox
has to trust an aios-controlled CA. This module is that CA.

**Zero stored state.** The CA private key is an HKDF subkey of
``AIOS_VAULT_KEY`` (domain context :data:`EGRESS_CA_HKDF_INFO`), so every
worker sharing the vault key derives the *same* keypair — no table, no
bootstrap race, no key file. The self-signed cert is regenerated per
process: its serial and validity window vary, but the subject DN and
keypair are deterministic, and trust anchors match on subject + public
key, so a leaf minted by one worker process chains against the cert copy
any other process installed.

Consequences operators should know:

* **Blast radius**: anyone holding ``AIOS_VAULT_KEY`` can re-derive this
  CA offline and mint leaf certs every aios sandbox trusts — a vault-key
  compromise grants active TLS interception of sandbox egress, not just
  at-rest decryption.
* **Rotation rides the vault key** (#858): the CA cannot rotate
  independently, and a rotation takes effect only after a worker
  restart (the per-seed cache below). No overlap window exists yet —
  the install step writes exactly one cert; #858 can add one by
  deriving from the previous key too (``update-ca-certificates`` and
  ``NODE_EXTRA_CA_CERTS`` both accept multi-PEM files).
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
from cryptography.x509.oid import NameOID

from aios.harness import runtime

# HKDF domain-separation context for the CA seed; bump the suffix if the
# derivation scheme ever changes incompatibly.
EGRESS_CA_HKDF_INFO = "aios-egress-ca-v1"

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


def get_egress_ca() -> EgressCA:
    """The worker's egress CA, derived from the current vault master key.

    Cached per seed (not per process), so a test or future rotation path
    that swaps ``runtime.crypto_box`` gets a matching CA without any
    cache-clearing choreography.
    """
    return _egress_ca_from_seed(
        runtime.require_crypto_box().derive_subkey_bytes(EGRESS_CA_HKDF_INFO)
    )


__all__ = [
    "CA_CERT_SANDBOX_PATH",
    "EGRESS_CA_HKDF_INFO",
    "SYSTEM_CA_BUNDLE_PATH",
    "TRUST_STORE_ENV",
    "EgressCA",
    "get_egress_ca",
]
