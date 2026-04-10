"""Tests for the libsodium credential vault."""

from __future__ import annotations

import base64
import os

import pytest

from aios.crypto.vault import KEY_BYTES, NONCE_BYTES, EncryptedBlob, Vault
from aios.errors import CredentialDecryptError


@pytest.fixture
def master_key() -> bytes:
    return os.urandom(KEY_BYTES)


@pytest.fixture
def vault(master_key: bytes) -> Vault:
    return Vault(master_key)


class TestRoundTrip:
    def test_encrypt_then_decrypt_recovers_plaintext(self, vault: Vault) -> None:
        original = "sk-ant-api03-Th1s1sN0tArealK3y"
        blob = vault.encrypt(original)
        recovered = vault.decrypt(blob)
        assert recovered == original

    def test_each_encryption_uses_a_fresh_nonce(self, vault: Vault) -> None:
        plaintext = "same secret"
        blob_a = vault.encrypt(plaintext)
        blob_b = vault.encrypt(plaintext)
        assert blob_a.nonce != blob_b.nonce
        assert blob_a.ciphertext != blob_b.ciphertext
        # but both decrypt to the same plaintext
        assert vault.decrypt(blob_a) == plaintext
        assert vault.decrypt(blob_b) == plaintext

    def test_unicode_plaintext_round_trips(self, vault: Vault) -> None:
        original = "sk-test-😀-ünïcödé-key"
        blob = vault.encrypt(original)
        assert vault.decrypt(blob) == original


class TestNonceAndKeyShape:
    def test_nonce_is_correct_length(self, vault: Vault) -> None:
        blob = vault.encrypt("anything")
        assert len(blob.nonce) == NONCE_BYTES

    def test_wrong_key_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 32 bytes"):
            Vault(b"too-short")

    def test_from_base64_loads_master_key(self) -> None:
        key = os.urandom(KEY_BYTES)
        encoded = base64.b64encode(key).decode("ascii")
        vault = Vault.from_base64(encoded)
        # Encryption should work after loading.
        blob = vault.encrypt("sentinel")
        assert vault.decrypt(blob) == "sentinel"

    def test_from_base64_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="not valid base64"):
            Vault.from_base64("not-base64-!@#$")


class TestTamperingAndKeyMismatch:
    def test_decrypt_with_different_key_raises_credential_error(self, vault: Vault) -> None:
        blob = vault.encrypt("real secret")
        other_vault = Vault(os.urandom(KEY_BYTES))
        with pytest.raises(CredentialDecryptError):
            other_vault.decrypt(blob)

    def test_tampered_ciphertext_raises_credential_error(self, vault: Vault) -> None:
        blob = vault.encrypt("real secret")
        tampered = EncryptedBlob(
            ciphertext=blob.ciphertext[:-1] + bytes([blob.ciphertext[-1] ^ 0x01]),
            nonce=blob.nonce,
        )
        with pytest.raises(CredentialDecryptError):
            vault.decrypt(tampered)

    def test_wrong_nonce_raises_credential_error(self, vault: Vault) -> None:
        blob = vault.encrypt("real secret")
        bad = EncryptedBlob(
            ciphertext=blob.ciphertext,
            nonce=os.urandom(NONCE_BYTES),
        )
        with pytest.raises(CredentialDecryptError):
            vault.decrypt(bad)
