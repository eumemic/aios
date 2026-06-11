"""The deterministic egress CA (``sandbox/egress_ca.py``).

What matters here, in order of load-bearingness:

- The keypair is a pure function of the seed — that's the whole
  zero-state design: every worker sharing ``AIOS_VAULT_KEY`` must derive
  the SAME CA, or leaf certs minted by one worker won't verify inside a
  sandbox whose trust store was installed by another.
- A leaf signed by one ``EgressCA`` instance chains against a *fresh*
  instance's cert (different serial / validity window) — the
  cross-process contract that makes per-process cert regeneration sound.
- ``install_egress_ca`` ships the exact command shape that works under
  ``bash -c``: ``printf '%s'`` (a bare ``printf '<PEM>'`` chokes on the
  leading ``-----BEGIN`` as an option string) and an ``&&`` chain so a
  partial install reports nonzero.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.x509.verification import PolicyBuilder, Store

from aios.crypto.vault import CryptoBox
from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult
from aios.sandbox.egress_ca import (
    CA_CERT_SANDBOX_PATH,
    SYSTEM_CA_BUNDLE_PATH,
    TRUST_STORE_ENV,
    EgressCA,
    get_egress_ca,
)
from aios.sandbox.setup import install_egress_ca
from aios.sandbox.spec import _assemble_plan
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


@pytest.fixture
def crypto_box_runtime() -> Iterator[CryptoBox]:
    prev = runtime.crypto_box
    runtime.crypto_box = CryptoBox(SEED_A)
    try:
        yield runtime.crypto_box
    finally:
        runtime.crypto_box = prev


class TestGetEgressCA:
    def test_follows_the_current_vault_key(self, crypto_box_runtime: CryptoBox) -> None:
        ca_a = get_egress_ca()
        assert get_egress_ca() is ca_a  # cached per seed
        runtime.crypto_box = CryptoBox(SEED_B)
        ca_b = get_egress_ca()
        assert ca_b is not ca_a
        assert ca_b.private_key.private_numbers() != ca_a.private_key.private_numbers()


class TestInstallEgressCA:
    async def test_command_shape(self, crypto_box_runtime: CryptoBox) -> None:
        backend = FakeBackend()
        handle = make_handle()

        await install_egress_ca(backend, handle)

        (call,) = [c[1] for c in backend.calls if c[0] == "exec"]
        cmd = call["command"]
        # printf '%s' is load-bearing: the PEM's leading ``-----BEGIN``
        # would otherwise be parsed as a printf option under bash -c.
        assert "printf '%s' '" in cmd
        assert "-----BEGIN CERTIFICATE-----" in cmd
        assert f"> {CA_CERT_SANDBOX_PATH} && update-ca-certificates" in cmd
        assert cmd.startswith("mkdir -p /usr/local/share/ca-certificates && ")

    async def test_failure_is_logged_not_raised(self, crypto_box_runtime: CryptoBox) -> None:
        backend = FakeBackend()
        backend.next_result = CommandResult(
            exit_code=1, stdout="", stderr="boom", timed_out=False, truncated=False
        )
        await install_egress_ca(backend, make_handle())


class TestTrustStoreEnvOnPlan:
    def test_plan_env_points_at_debian_bundle_paths(self) -> None:
        with (
            patch(
                "aios.sandbox.volumes.ensure_session_attachments_dir",
                return_value=Path("/tmp/a"),
            ),
            patch("aios.sandbox.volumes.ensure_session_uploads_dir", return_value=Path("/tmp/u")),
        ):
            plan = _assemble_plan(
                session_id="sess_01TEST",
                instance_id="inst_TEST",
                image="aios-sandbox:test",
                workspace_path=Path("/tmp/w"),
                env_config=None,
                session_env={},
                memory_echoes=[],
                github_echoes=[],
                git_proxy=None,
                tool_broker_url="http://aios-worker:54321",
                tool_broker_secret="secret123",
                tool_socket_host_path=None,
            )
        env = plan.spec.environment
        assert env["SSL_CERT_FILE"] == SYSTEM_CA_BUNDLE_PATH
        assert env["REQUESTS_CA_BUNDLE"] == SYSTEM_CA_BUNDLE_PATH
        assert env["NODE_EXTRA_CA_CERTS"] == CA_CERT_SANDBOX_PATH
        assert set(TRUST_STORE_ENV) <= set(env)
