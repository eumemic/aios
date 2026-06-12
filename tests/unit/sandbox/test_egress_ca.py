"""The deterministic egress CA (``sandbox/egress_ca.py``).

What matters here, in order of load-bearingness:

- The keypair is a pure function of the seed — that's the whole
  zero-state design: every worker sharing ``AIOS_EGRESS_CA_KEY`` must derive
  the SAME CA, or leaf certs minted by one worker won't verify inside a
  sandbox whose trust store was installed by another.
- A leaf signed by one ``EgressCA`` instance chains against a *fresh*
  instance's cert (different serial / validity window) — the
  cross-process contract that makes per-process cert regeneration sound.
- ``install_egress_ca`` ships the exact command shape that works under
  ``bash -c`` (rationale in its docstring).
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.x509.verification import PolicyBuilder, Store

from aios.crypto.vault import CryptoBox
from aios.harness import runtime
from aios.models.environments import EnvironmentConfig
from aios.sandbox.backends.base import CommandResult
from aios.sandbox.egress_ca import (
    CA_CERT_SANDBOX_PATH,
    LEAF_VALIDITY_DAYS,
    SYSTEM_CA_BUNDLE_PATH,
    EgressCA,
    get_egress_ca,
    mint_server_leaf,
)
from aios.sandbox.setup import install_egress_ca
from aios.sandbox.spec import ProvisioningPlan, _assemble_plan
from tests.helpers.sandbox import FakeBackend, make_handle
from tests.helpers.tls import mint_leaf

SEED_A = bytes(range(32))
SEED_B = bytes(range(1, 33))


class TestDeterminism:
    def test_same_seed_same_keypair_and_ski(self) -> None:
        a, b = EgressCA(SEED_A), EgressCA(SEED_A)
        assert a.private_key.private_numbers() == b.private_key.private_numbers()
        ski_a = a.certificate.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
        ski_b = b.certificate.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
        assert ski_a.value.digest == ski_b.value.digest
        # The cert bytes themselves differ (random serial) — only the
        # subject + key are the cross-process contract.
        assert a.certificate.serial_number != b.certificate.serial_number

    def test_different_seed_different_keypair(self) -> None:
        assert (
            EgressCA(SEED_A).private_key.private_numbers()
            != EgressCA(SEED_B).private_key.private_numbers()
        )


class TestCertificate:
    def test_ca_profile(self) -> None:
        cert = EgressCA(SEED_A).certificate
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
        assert bc.critical
        assert bc.value.ca is True
        assert bc.value.path_length == 0
        ku = cert.extensions.get_extension_for_class(x509.KeyUsage)
        assert ku.critical
        assert ku.value.key_cert_sign and ku.value.crl_sign
        assert not ku.value.digital_signature
        assert cert.subject == cert.issuer
        cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
        assert cn == "aios Egress CA"

    def test_leaf_minted_by_one_instance_chains_against_a_fresh_instance(self) -> None:
        """The cross-process story: worker A installed its cert copy into
        the sandbox; worker B (same seed, different serial/notBefore)
        mints the leaf. Verification must succeed."""
        leaf, _ = mint_leaf(EgressCA(SEED_A), "api.example.com")
        fresh_ca_cert = EgressCA(SEED_A).certificate
        verifier = (
            PolicyBuilder()
            .store(Store([fresh_ca_cert]))
            .build_server_verifier(x509.DNSName("api.example.com"))
        )
        chain = verifier.verify(leaf, [])
        assert chain[-1] == fresh_ca_cert

    def test_leaf_from_different_seed_does_not_verify(self) -> None:
        leaf, _ = mint_leaf(EgressCA(SEED_B), "api.example.com")
        verifier = (
            PolicyBuilder()
            .store(Store([EgressCA(SEED_A).certificate]))
            .build_server_verifier(x509.DNSName("api.example.com"))
        )
        with pytest.raises(x509.verification.VerificationError):
            verifier.verify(leaf, [])


class TestMintServerLeaf:
    """The production leaf contract (the single authoritative mint the egress
    proxy and the tls test helper both call)."""

    def test_leaf_contract(self) -> None:
        cert, key = mint_server_leaf(EgressCA(SEED_A), "api.example.com")

        # keyid-only AKID: the SKI matches across worker processes; an
        # issuer+serial reference would break chains (the CA serial varies).
        akid = cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value
        assert akid.key_identifier is not None
        assert akid.authority_cert_serial_number is None
        assert akid.authority_cert_issuer is None

        # Not a CA; critical DNS SAN as the authoritative identity; empty
        # subject (no CN — avoids the 64-char ub-common-name limit; valid
        # precisely because the SAN is critical).
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        assert bc.ca is False
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        assert san.critical is True
        assert san.value.get_values_for_type(x509.DNSName) == ["api.example.com"]
        assert list(cert.subject) == []

        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku
        # The leaf key signs the handshake — it is a fresh per-leaf key, not
        # the CA's.
        assert key.public_key().public_numbers() == (
            cert.public_key().public_numbers()  # type: ignore[union-attr]
        )

    def test_short_validity(self) -> None:
        cert, _ = mint_server_leaf(EgressCA(SEED_A), "api.example.com")
        span = cert.not_valid_after_utc - cert.not_valid_before_utc
        # LEAF_VALIDITY_DAYS plus the one-hour backdate.
        assert span == timedelta(days=LEAF_VALIDITY_DAYS, hours=1)

    def test_long_host_mints_and_chains(self) -> None:
        # A 65-253-char host is valid per the credential validator but exceeds
        # the 64-char CN limit — it must still mint (SAN-only) and chain.
        host = ("a" * 61) + ".example.com"  # 73 chars
        assert len(host) > 64
        leaf, _ = mint_server_leaf(EgressCA(SEED_A), host)
        fresh_ca = EgressCA(SEED_A)  # same key, different serial (cross-process)
        verifier = (
            PolicyBuilder()
            .store(Store([fresh_ca.certificate]))
            .build_server_verifier(x509.DNSName(host))
        )
        assert verifier.verify(leaf, [])[-1] == fresh_ca.certificate


@pytest.fixture
def crypto_box_runtime() -> Iterator[CryptoBox]:
    prev = runtime.crypto_box
    runtime.crypto_box = CryptoBox(SEED_A)
    try:
        yield runtime.crypto_box
    finally:
        runtime.crypto_box = prev


class TestGetEgressCA:
    def test_uses_egress_ca_key_not_vault_key(
        self, crypto_box_runtime: CryptoBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aios.config import get_settings

        monkeypatch.setenv("AIOS_EGRESS_CA_KEY", base64.b64encode(SEED_A).decode("ascii"))
        get_settings.cache_clear()
        ca_a = get_egress_ca()
        assert get_egress_ca() is ca_a  # cached per seed
        runtime.crypto_box = CryptoBox(SEED_B)
        assert get_egress_ca().private_key.private_numbers() == ca_a.private_key.private_numbers()

        monkeypatch.setenv("AIOS_EGRESS_CA_KEY", base64.b64encode(SEED_B).decode("ascii"))
        get_settings.cache_clear()
        ca_b = get_egress_ca()
        assert ca_b.private_key.private_numbers() != ca_a.private_key.private_numbers()


class TestInstallEgressCA:
    async def test_command_shape(self, crypto_box_runtime: CryptoBox) -> None:
        backend = FakeBackend()
        handle = make_handle()

        await install_egress_ca(backend, handle)

        (call,) = [c[1] for c in backend.calls if c[0] == "exec"]
        cmd = call["command"]
        # printf '%s' is load-bearing, and the PEM must be its quoted
        # ARGUMENT (adjacency, not mere co-presence): the PEM's leading
        # ``-----BEGIN`` parsed as printf's format would be an invalid
        # option under bash -c.
        assert "printf '%s' '-----BEGIN CERTIFICATE-----" in cmd
        assert f"> {CA_CERT_SANDBOX_PATH} && update-ca-certificates" in cmd
        assert cmd.startswith("mkdir -p /usr/local/share/ca-certificates && ")

    async def test_failure_is_logged_not_raised(self, crypto_box_runtime: CryptoBox) -> None:
        backend = FakeBackend()
        backend.next_result = CommandResult(
            exit_code=1, stdout="", stderr="boom", timed_out=False, truncated=False
        )

        await install_egress_ca(backend, make_handle())

        # The install was attempted (one exec) and the failure stayed a
        # warning — no raise reached us.
        assert [c[0] for c in backend.calls] == ["exec"]


def _assemble_default_plan(
    env_config: EnvironmentConfig | None = None, image: str = "aios-sandbox:test"
) -> ProvisioningPlan:
    with (
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/a"),
        ),
        patch("aios.sandbox.volumes.ensure_session_uploads_dir", return_value=Path("/tmp/u")),
    ):
        return _assemble_plan(
            session_id="sess_01TEST",
            instance_id="inst_TEST",
            image=image,
            workspace_path=Path("/tmp/w"),
            env_config=env_config,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            tool_broker_url="http://aios-worker:54321",
            tool_broker_secret="secret123",
            tool_socket_host_path=None,
        )


class TestTrustStoreEnvOnPlan:
    def test_plan_env_points_at_debian_bundle_paths(self) -> None:
        plan = _assemble_default_plan()
        env = plan.spec.environment
        assert env["SSL_CERT_FILE"] == SYSTEM_CA_BUNDLE_PATH
        assert env["REQUESTS_CA_BUNDLE"] == SYSTEM_CA_BUNDLE_PATH
        assert env["NODE_EXTRA_CA_CERTS"] == CA_CERT_SANDBOX_PATH

    def test_environment_env_overrides_trust_store_defaults(self) -> None:
        """The documented escape hatch for custom images (#724) with a
        non-Debian trust-store layout: an environment's ``env`` wins
        over the trust-store defaults in the merge order."""
        plan = _assemble_default_plan(
            env_config=EnvironmentConfig(env={"SSL_CERT_FILE": "/etc/pki/tls/certs/ca-bundle.crt"}),
            image="custom-image:rhel",
        )
        env = plan.spec.environment
        assert env["SSL_CERT_FILE"] == "/etc/pki/tls/certs/ca-bundle.crt"
        assert env["REQUESTS_CA_BUNDLE"] == SYSTEM_CA_BUNDLE_PATH  # untouched default
