"""Tests for the libsodium CryptoBox."""

from __future__ import annotations

import base64
import os

import pytest

from aios.crypto.vault import KEY_BYTES, NONCE_BYTES, CryptoBox, EncryptedBlob
from aios.errors import CryptoDecryptError


@pytest.fixture
def master_key() -> bytes:
    return os.urandom(KEY_BYTES)


@pytest.fixture
def crypto_box(master_key: bytes) -> CryptoBox:
    return CryptoBox(master_key)


class TestRoundTrip:
    def test_encrypt_then_decrypt_recovers_plaintext(self, crypto_box: CryptoBox) -> None:
        original = "sk-ant-api03-Th1s1sN0tArealK3y"
        blob = crypto_box.encrypt(original)
        recovered = crypto_box.decrypt(blob)
        assert recovered == original

    def test_each_encryption_uses_a_fresh_nonce(self, crypto_box: CryptoBox) -> None:
        plaintext = "same secret"
        blob_a = crypto_box.encrypt(plaintext)
        blob_b = crypto_box.encrypt(plaintext)
        assert blob_a.nonce != blob_b.nonce
        assert blob_a.ciphertext != blob_b.ciphertext
        # but both decrypt to the same plaintext
        assert crypto_box.decrypt(blob_a) == plaintext
        assert crypto_box.decrypt(blob_b) == plaintext

    def test_unicode_plaintext_round_trips(self, crypto_box: CryptoBox) -> None:
        original = "sk-test-😀-ünïcödé-key"
        blob = crypto_box.encrypt(original)
        assert crypto_box.decrypt(blob) == original


class TestNonceAndKeyShape:
    def test_nonce_is_correct_length(self, crypto_box: CryptoBox) -> None:
        blob = crypto_box.encrypt("anything")
        assert len(blob.nonce) == NONCE_BYTES

    def test_wrong_key_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 32 bytes"):
            CryptoBox(b"too-short")

    def test_from_base64_loads_master_key(self) -> None:
        key = os.urandom(KEY_BYTES)
        encoded = base64.b64encode(key).decode("ascii")
        crypto_box = CryptoBox.from_base64(encoded)
        # Encryption should work after loading.
        blob = crypto_box.encrypt("sentinel")
        assert crypto_box.decrypt(blob) == "sentinel"

    def test_from_base64_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="not valid base64"):
            CryptoBox.from_base64("not-base64-!@#$")


class TestTamperingAndKeyMismatch:
    def test_decrypt_with_different_key_raises_credential_error(
        self, crypto_box: CryptoBox
    ) -> None:
        blob = crypto_box.encrypt("real secret")
        other_crypto_box = CryptoBox(os.urandom(KEY_BYTES))
        with pytest.raises(CryptoDecryptError):
            other_crypto_box.decrypt(blob)

    def test_tampered_ciphertext_raises_credential_error(self, crypto_box: CryptoBox) -> None:
        blob = crypto_box.encrypt("real secret")
        tampered = EncryptedBlob(
            ciphertext=blob.ciphertext[:-1] + bytes([blob.ciphertext[-1] ^ 0x01]),
            nonce=blob.nonce,
        )
        with pytest.raises(CryptoDecryptError):
            crypto_box.decrypt(tampered)

    def test_wrong_nonce_raises_credential_error(self, crypto_box: CryptoBox) -> None:
        blob = crypto_box.encrypt("real secret")
        bad = EncryptedBlob(
            ciphertext=blob.ciphertext,
            nonce=os.urandom(NONCE_BYTES),
        )
        with pytest.raises(CryptoDecryptError):
            crypto_box.decrypt(bad)
