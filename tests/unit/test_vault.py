"""Tests for the libsodium CryptoBox and vault service-layer helpers."""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from aios.crypto.vault import KEY_BYTES, NONCE_BYTES, CryptoBox, EncryptedBlob
from aios.errors import CryptoDecryptError
from aios.models.vaults import (
    TokenEndpointAuthBasic,
    TokenEndpointAuthNone,
    TokenEndpointAuthPost,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
)
from aios.services import vaults as vaults_service
from aios.services.vaults import _extract_auth_payload, _merge_auth_payload


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


# ── Service-layer helpers ────────────────────────────────────────────────────


def _oauth_create(**overrides: Any) -> VaultCredentialCreate:
    base = {
        "mcp_server_url": "https://mcp.example.com",
        "auth_type": "mcp_oauth",
        "access_token": SecretStr("at"),
        "client_id": "cid",
        "token_endpoint": "https://issuer.example/token",
    }
    base.update(overrides)
    return VaultCredentialCreate(**base)


class TestExtractAuthPayload:
    def test_static_bearer_only_token(self) -> None:
        body = VaultCredentialCreate(
            mcp_server_url="https://x.com",
            auth_type="static_bearer",
            token=SecretStr("hello"),
        )
        payload = _extract_auth_payload(body)
        assert payload == {"token": "hello"}

    def test_oauth_serializes_token_endpoint_auth_basic(self) -> None:
        body = _oauth_create(
            token_endpoint_auth=TokenEndpointAuthBasic(
                method="client_secret_basic",
                client_secret=SecretStr("shh"),
            ),
        )
        payload = _extract_auth_payload(body)
        assert payload["token_endpoint_auth"] == {
            "method": "client_secret_basic",
            "client_secret": "shh",
        }
        assert payload["access_token"] == "at"
        assert payload["client_id"] == "cid"

    def test_oauth_serializes_token_endpoint_auth_post(self) -> None:
        body = _oauth_create(
            token_endpoint_auth=TokenEndpointAuthPost(
                method="client_secret_post",
                client_secret=SecretStr("shh"),
            ),
        )
        payload = _extract_auth_payload(body)
        assert payload["token_endpoint_auth"] == {
            "method": "client_secret_post",
            "client_secret": "shh",
        }

    def test_oauth_serializes_token_endpoint_auth_none(self) -> None:
        body = _oauth_create(
            token_endpoint_auth=TokenEndpointAuthNone(method="none"),
        )
        payload = _extract_auth_payload(body)
        assert payload["token_endpoint_auth"] == {"method": "none"}
        # ensure no client_secret leaked into the payload
        assert "client_secret" not in payload["token_endpoint_auth"]

    def test_oauth_omits_token_endpoint_auth_when_not_provided(self) -> None:
        body = _oauth_create()
        payload = _extract_auth_payload(body)
        assert "token_endpoint_auth" not in payload

    def test_payload_round_trips_through_json(self) -> None:
        # The whole point of _extract_auth_payload is that the result is
        # JSON-serializable for storage in the encrypted blob.
        body = _oauth_create(
            token_endpoint_auth=TokenEndpointAuthBasic(
                method="client_secret_basic",
                client_secret=SecretStr("shh"),
            ),
        )
        payload = _extract_auth_payload(body)
        json.dumps(payload)  # would raise TypeError if not JSON-able


class TestMergeAuthPayload:
    def test_swaps_method_basic_to_post(self) -> None:
        existing = {
            "access_token": "at",
            "token_endpoint_auth": {
                "method": "client_secret_basic",
                "client_secret": "old",
            },
        }
        update = VaultCredentialUpdate(
            token_endpoint_auth=TokenEndpointAuthPost(
                method="client_secret_post",
                client_secret=SecretStr("rotated"),
            ),
        )
        merged = _merge_auth_payload(existing, update, "mcp_oauth")
        assert merged["token_endpoint_auth"] == {
            "method": "client_secret_post",
            "client_secret": "rotated",
        }
        assert merged["access_token"] == "at"  # untouched

    def test_preserves_existing_when_field_omitted(self) -> None:
        existing = {
            "access_token": "at",
            "token_endpoint_auth": {"method": "none"},
        }
        update = VaultCredentialUpdate()  # nothing set
        merged = _merge_auth_payload(existing, update, "mcp_oauth")
        assert merged == existing

    def test_unsets_field_when_set_to_none(self) -> None:
        existing = {"access_token": "at", "client_id": "cid"}
        update = VaultCredentialUpdate(client_id=None)
        merged = _merge_auth_payload(existing, update, "mcp_oauth")
        assert "client_id" not in merged
        assert merged["access_token"] == "at"


# ── update_vault_credential: no _UNSET sentinel leaks into queries ───────────


def _fake_pool_yielding_conn(conn: Any) -> Any:
    """Build a stand-in for asyncpg.Pool where ``async with pool.acquire()`` yields *conn*."""
    pool = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire_cm
    return pool


def _existing_credential() -> VaultCredential:
    from datetime import UTC, datetime

    return VaultCredential(
        id="vc_1",
        vault_id="vlt_1",
        display_name="orig",
        mcp_server_url="https://mcp.example.com",
        auth_type="mcp_oauth",
        metadata={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class TestUpdateVaultCredentialCallSite:
    """The service must build kwargs from ``model_fields_set`` and pass ``...``
    (Ellipsis) for fields the client did not explicitly set — never reach into
    a private query-layer sentinel."""

    @pytest.mark.asyncio
    async def test_omits_display_name_when_not_in_fields_set(self, crypto_box: CryptoBox) -> None:
        existing = _existing_credential()
        existing_blob = crypto_box.encrypt(json.dumps({"access_token": "at"}))
        conn = MagicMock()
        pool = _fake_pool_yielding_conn(conn)

        with (
            patch.object(
                vaults_service.queries,
                "get_vault_credential_with_blob",
                AsyncMock(return_value=(existing, existing_blob)),
            ),
            patch.object(
                vaults_service.queries,
                "update_vault_credential",
                AsyncMock(return_value=existing),
            ) as upd,
        ):
            body = VaultCredentialUpdate()  # nothing set
            await vaults_service.update_vault_credential(
                pool,
                crypto_box,
                vault_id="vlt_1",
                credential_id="vc_1",
                body=body,
            )

        upd.assert_awaited_once()
        kwargs = upd.await_args.kwargs
        assert kwargs["display_name"] is ...
        assert kwargs["metadata"] is ...
        assert kwargs["blob"] is not None  # always re-encrypted

    @pytest.mark.asyncio
    async def test_passes_display_name_when_set_even_to_none(self, crypto_box: CryptoBox) -> None:
        existing = _existing_credential()
        existing_blob = crypto_box.encrypt(json.dumps({"access_token": "at"}))
        conn = MagicMock()
        pool = _fake_pool_yielding_conn(conn)

        with (
            patch.object(
                vaults_service.queries,
                "get_vault_credential_with_blob",
                AsyncMock(return_value=(existing, existing_blob)),
            ),
            patch.object(
                vaults_service.queries,
                "update_vault_credential",
                AsyncMock(return_value=existing),
            ) as upd,
        ):
            body = VaultCredentialUpdate(display_name=None)  # explicitly set to None
            await vaults_service.update_vault_credential(
                pool,
                crypto_box,
                vault_id="vlt_1",
                credential_id="vc_1",
                body=body,
            )

        kwargs = upd.await_args.kwargs
        assert kwargs["display_name"] is None  # not Ellipsis — explicitly passed
