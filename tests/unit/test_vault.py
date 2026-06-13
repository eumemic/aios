"""Tests for the libsodium CryptoBox and vault service-layer helpers."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nacl.secret import SecretBox
from pydantic import SecretStr

from aios.crypto.vault import BLOB_VERSION, KEY_BYTES, NONCE_BYTES, CryptoBox, EncryptedBlob
from aios.db import queries
from aios.db.queries import EnvVarCredentialRow
from aios.errors import CryptoDecryptError, OAuthRefreshError, ValidationError
from aios.models.environments import EnvironmentConfig, UnrestrictedNetworking
from aios.models.vaults import (
    TokenEndpointAuthBasic,
    TokenEndpointAuthNone,
    TokenEndpointAuthPost,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
)
from aios.services import sessions as sessions_service
from aios.services import vaults as vaults_service
from aios.services.vaults import (
    REFRESH_SKEW_SECONDS,
    SECRET_PLACEHOLDER_PREFIX,
    _extract_auth_payload,
    _merge_auth_payload,
    env_var_credential_containment_error,
    is_expiring,
    mint_secret_placeholder,
    refresh_credential,
)
from tests.helpers.sandbox import limited_env
from tests.unit.conftest import fake_pool_yielding_conn


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


class TestLegacyVersionByteCollision:
    """#858 R4: legacy unversioned blobs must round-trip — including the
    ~1/256 whose leading MAC byte coincidentally equals ``BLOB_VERSION``.

    Before the decrypt-retry fix, such a blob had its real first byte
    mis-stripped as a version prefix and decryption failed permanently —
    a silent data-loss class on otherwise-healthy rows.
    """

    _KEY = bytes(range(KEY_BYTES))

    @classmethod
    def _legacy_blob_colliding_with_version_byte(cls, plaintext: bytes) -> EncryptedBlob:
        """Brute-force a nonce until the raw SecretBox ciphertext (MAC-first)
        genuinely begins with the version byte. ~256 deterministic attempts."""
        raw_box = SecretBox(cls._KEY)
        for i in range(100_000):
            nonce = i.to_bytes(NONCE_BYTES, "big")
            ciphertext = raw_box.encrypt(plaintext, nonce).ciphertext
            if ciphertext[:1] == BLOB_VERSION:
                return EncryptedBlob(ciphertext=ciphertext, nonce=nonce)
        raise AssertionError("no version-byte-colliding nonce found in 100k attempts")

    def test_legacy_blob_with_leading_version_byte_round_trips(self) -> None:
        blob = self._legacy_blob_colliding_with_version_byte(b"legacy-collision")
        assert blob.ciphertext[:1] == BLOB_VERSION  # the trap: looks versioned
        assert CryptoBox(self._KEY).decrypt(blob) == "legacy-collision"

    def test_versioned_blob_still_decrypts(self) -> None:
        box = CryptoBox(self._KEY)
        blob = box.encrypt("versioned-payload")
        assert blob.ciphertext[:1] == BLOB_VERSION
        assert box.decrypt(blob) == "versioned-payload"

    def test_corrupted_colliding_blob_still_raises(self) -> None:
        blob = self._legacy_blob_colliding_with_version_byte(b"legacy-collision")
        corrupted = EncryptedBlob(
            ciphertext=blob.ciphertext[:-1] + bytes([blob.ciphertext[-1] ^ 0x01]),
            nonce=blob.nonce,
        )
        with pytest.raises(CryptoDecryptError):
            CryptoBox(self._KEY).decrypt(corrupted)

    def test_corrupted_versioned_blob_still_raises(self) -> None:
        box = CryptoBox(self._KEY)
        blob = box.encrypt("versioned-payload")
        corrupted = EncryptedBlob(
            ciphertext=blob.ciphertext[:-1] + bytes([blob.ciphertext[-1] ^ 0x01]),
            nonce=blob.nonce,
        )
        with pytest.raises(CryptoDecryptError):
            box.decrypt(corrupted)


# ── Service-layer helpers ────────────────────────────────────────────────────


def _oauth_create(**overrides: Any) -> VaultCredentialCreate:
    base = {
        "target_url": "https://mcp.example.com",
        "auth_type": "oauth2_refresh",
        "access_token": SecretStr("at"),
        "client_id": "cid",
        "token_endpoint": "https://issuer.example/token",
    }
    base.update(overrides)
    return VaultCredentialCreate(**base)


class TestExtractAuthPayload:
    def test_bearer_header_only_token(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://x.com",
            auth_type="bearer_header",
            token=SecretStr("hello"),
        )
        payload = _extract_auth_payload(body)
        assert payload == {"token": "hello"}

    def test_basic_username_and_password(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://x.com",
            auth_type="basic",
            username=SecretStr("alice"),
            password=SecretStr("s3cret"),
        )
        payload = _extract_auth_payload(body)
        assert payload == {"username": "alice", "password": "s3cret"}

    def test_basic_requires_username(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://x.com",
            auth_type="basic",
            password=SecretStr("s3cret"),
        )
        with pytest.raises(ValidationError):
            _extract_auth_payload(body)

    def test_basic_requires_password(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://x.com",
            auth_type="basic",
            username=SecretStr("alice"),
        )
        with pytest.raises(ValidationError):
            _extract_auth_payload(body)

    def test_custom_header_name_and_value(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://api.example.com",
            auth_type="custom_header",
            header_name="X-Api-Key",
            header_value=SecretStr("bu_secret"),
        )
        payload = _extract_auth_payload(body)
        assert payload == {"header_name": "X-Api-Key", "header_value": "bu_secret"}

    def test_environment_variable_only_secret_value(self) -> None:
        body = VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name="GITHUB_TOKEN",
            allowed_hosts=["api.github.com"],
            secret_value=SecretStr("ghp_xxx"),
        )
        payload = _extract_auth_payload(body)
        # Only the secret value lands in the encrypted blob; secret_name /
        # allowed_hosts are plaintext columns, not part of the payload.
        assert payload == {"secret_value": "ghp_xxx"}

    def test_environment_variable_requires_secret_value(self) -> None:
        body = VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name="GITHUB_TOKEN",
            allowed_hosts=["api.github.com"],
        )
        with pytest.raises(ValidationError):
            _extract_auth_payload(body)

    def test_custom_header_requires_header_name(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://api.example.com",
            auth_type="custom_header",
            header_value=SecretStr("bu_secret"),
        )
        with pytest.raises(ValidationError):
            _extract_auth_payload(body)

    def test_custom_header_requires_header_value(self) -> None:
        body = VaultCredentialCreate(
            target_url="https://api.example.com",
            auth_type="custom_header",
            header_name="X-Api-Key",
        )
        with pytest.raises(ValidationError):
            _extract_auth_payload(body)

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
        merged = _merge_auth_payload(existing, update, "oauth2_refresh")
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
        merged = _merge_auth_payload(existing, update, "oauth2_refresh")
        assert merged == existing

    def test_unsets_field_when_set_to_none(self) -> None:
        existing = {"access_token": "at", "client_id": "cid"}
        update = VaultCredentialUpdate(client_id=None)
        merged = _merge_auth_payload(existing, update, "oauth2_refresh")
        assert "client_id" not in merged
        assert merged["access_token"] == "at"

    def test_unsetting_required_bearer_token_is_rejected(self) -> None:
        """A PUT with ``{"token": null}`` on a bearer_header credential
        previously silently dropped the token from the merged payload,
        leaving an "active" credential with no token. Downstream
        rendering produced ``Authorization`` headers with an empty bearer
        token and the upstream reliably 401'd. Reject at the merge site
        so the operator gets a clear ``ValidationError`` instead of a
        silently-broken credential."""
        existing = {"token": "secret"}
        update = VaultCredentialUpdate(token=None)
        with pytest.raises(ValidationError, match="bearer_header"):
            _merge_auth_payload(existing, update, "bearer_header")

    def test_unsetting_required_basic_username_is_rejected(self) -> None:
        existing = {"username": "alice", "password": "secret"}
        update = VaultCredentialUpdate(username=None)
        with pytest.raises(ValidationError, match="basic"):
            _merge_auth_payload(existing, update, "basic")

    def test_unsetting_required_basic_password_is_rejected(self) -> None:
        existing = {"username": "alice", "password": "secret"}
        update = VaultCredentialUpdate(password=None)
        with pytest.raises(ValidationError, match="basic"):
            _merge_auth_payload(existing, update, "basic")

    def test_unsetting_required_custom_header_value_is_rejected(self) -> None:
        existing = {"header_name": "X-Api-Key", "header_value": "sekrit"}
        update = VaultCredentialUpdate(header_value=None)
        with pytest.raises(ValidationError, match="custom_header"):
            _merge_auth_payload(existing, update, "custom_header")

    def test_unsetting_required_oauth_access_token_is_rejected(self) -> None:
        existing = {"access_token": "at", "client_id": "cid"}
        update = VaultCredentialUpdate(access_token=None)
        with pytest.raises(ValidationError, match="oauth2_refresh"):
            _merge_auth_payload(existing, update, "oauth2_refresh")

    def test_environment_variable_rotates_secret_value(self) -> None:
        existing = {"secret_value": "old"}
        update = VaultCredentialUpdate(secret_value=SecretStr("new"))
        merged = _merge_auth_payload(existing, update, "environment_variable")
        assert merged == {"secret_value": "new"}

    def test_environment_variable_preserves_secret_on_omit(self) -> None:
        existing = {"secret_value": "keep"}
        merged = _merge_auth_payload(existing, VaultCredentialUpdate(), "environment_variable")
        assert merged == {"secret_value": "keep"}

    def test_unsetting_required_secret_value_is_rejected(self) -> None:
        existing = {"secret_value": "secret"}
        update = VaultCredentialUpdate(secret_value=None)
        with pytest.raises(ValidationError, match="environment_variable"):
            _merge_auth_payload(existing, update, "environment_variable")


# ── update_vault_credential: no private sentinel leaks into queries ──────────


def _existing_credential() -> VaultCredential:

    return VaultCredential(
        id="vc_1",
        vault_id="vlt_1",
        display_name="orig",
        target_url="https://mcp.example.com",
        auth_type="oauth2_refresh",
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
        account_id = "acc_test_stub"  # PR 3 scaffolding
        existing = _existing_credential()
        existing_blob = crypto_box.derive_account_subkey(account_id).encrypt(
            json.dumps({"access_token": "at"})
        )
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)

        with (
            patch.object(
                queries,
                "get_vault_credential_with_blob",
                AsyncMock(return_value=(existing, existing_blob)),
            ),
            patch.object(
                queries,
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
                account_id=account_id,
            )

        upd.assert_awaited_once()
        assert upd.await_args is not None
        kwargs = upd.await_args.kwargs
        assert kwargs["display_name"] is ...
        assert kwargs["metadata"] is ...
        assert kwargs["blob"] is not None  # always re-encrypted

    @pytest.mark.asyncio
    async def test_passes_display_name_when_set_even_to_none(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        existing = _existing_credential()
        existing_blob = crypto_box.derive_account_subkey(account_id).encrypt(
            json.dumps({"access_token": "at"})
        )
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)

        with (
            patch.object(
                queries,
                "get_vault_credential_with_blob",
                AsyncMock(return_value=(existing, existing_blob)),
            ),
            patch.object(
                queries,
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
                account_id=account_id,
            )

        assert upd.await_args is not None
        kwargs = upd.await_args.kwargs
        assert kwargs["display_name"] is None  # not Ellipsis — explicitly passed


# ── OAuth refresh ────────────────────────────────────────────────────────────


def _conn_with_transaction() -> MagicMock:
    """Return a MagicMock conn whose ``conn.transaction()`` is a working async CM."""
    conn = MagicMock()
    txn_cm = MagicMock()
    txn_cm.__aenter__ = AsyncMock(return_value=None)
    txn_cm.__aexit__ = AsyncMock(return_value=None)
    conn.transaction.return_value = txn_cm
    conn.execute = AsyncMock()
    return conn


def _http_response(*, status: int = 200, body: dict[str, Any] | None = None) -> MagicMock:
    """Build a fake httpx.Response with ``json()`` and ``raise_for_status()``."""
    import httpx as _httpx

    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=body or {})
    if status >= 400:

        def _raise() -> None:
            raise _httpx.HTTPStatusError(
                f"server returned {status}", request=MagicMock(), response=resp
            )

        resp.raise_for_status = MagicMock(side_effect=_raise)
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _async_client_returning(resp: Any) -> MagicMock:
    """Build a MagicMock standing in for ``httpx.AsyncClient(...)``."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=resp)
    return client


def _expiring_oauth_payload(**overrides: Any) -> dict[str, Any]:

    base = {
        "access_token": "old-at",
        "refresh_token": "rt-1",
        "client_id": "cid",
        "token_endpoint": "https://issuer.example/token",
        "expires_at": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
        "token_endpoint_auth": {"method": "none"},
    }
    base.update(overrides)
    return base


class TestIsExpiring:
    def test_far_future_is_not_expiring(self) -> None:

        payload = {"expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()}
        assert is_expiring(payload) is False

    def test_within_skew_is_expiring(self) -> None:

        # Inside the skew window (5 s < 30 s default).
        payload = {"expires_at": (datetime.now(UTC) + timedelta(seconds=5)).isoformat()}
        assert is_expiring(payload) is True

    def test_already_expired_is_expiring(self) -> None:

        payload = {"expires_at": (datetime.now(UTC) - timedelta(minutes=5)).isoformat()}
        assert is_expiring(payload) is True

    def test_missing_expires_at_is_not_expiring(self) -> None:
        # Treat absence as "never expires" — refresh path stays out of the way.
        assert is_expiring({}) is False

    def test_naive_datetime_assumed_utc(self) -> None:

        # Some providers return naive ISO strings; treat as UTC.
        future_naive = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=5)
        assert is_expiring({"expires_at": future_naive.isoformat()}) is True


class TestRefreshCredential:
    """Mocked-httpx unit tests for the locked refresh helper.

    The conn is a MagicMock; the lock query is patched to return a real
    decryptable blob so the function exercises decrypt → POST → re-encrypt
    → UPDATE end-to-end against fake I/O.
    """

    @pytest.mark.asyncio
    async def test_skips_when_token_not_expiring(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding

        payload = _expiring_oauth_payload(
            expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        )
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        client.post.assert_not_awaited()
        conn.execute.assert_not_awaited()  # row not updated

    @pytest.mark.asyncio
    async def test_basic_method_uses_basic_auth(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        import httpx as _httpx

        payload = _expiring_oauth_payload(
            token_endpoint_auth={"method": "client_secret_basic", "client_secret": "shh"},
        )
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        kwargs = client.post.await_args.kwargs
        assert isinstance(kwargs["auth"], _httpx.BasicAuth)
        # client_secret never leaks into the form body.
        assert "client_secret" not in kwargs["data"]
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "rt-1"

    @pytest.mark.asyncio
    async def test_post_method_includes_secret_in_body(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload(
            token_endpoint_auth={"method": "client_secret_post", "client_secret": "shh"},
        )
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        kwargs = client.post.await_args.kwargs
        assert "auth" not in kwargs
        assert kwargs["data"]["client_secret"] == "shh"
        assert kwargs["data"]["client_id"] == "cid"

    @pytest.mark.asyncio
    async def test_none_method_includes_only_client_id(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload(
            token_endpoint_auth={"method": "none"},
        )
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        kwargs = client.post.await_args.kwargs
        assert "auth" not in kwargs
        assert "client_secret" not in kwargs["data"]
        assert kwargs["data"]["client_id"] == "cid"

    @pytest.mark.asyncio
    async def test_string_expires_in_is_accepted(self, crypto_box: CryptoBox) -> None:
        """Some OAuth providers return ``expires_in`` as a JSON string ("3600").

        Without ``int()`` conversion the new token would be stored without an
        ``expires_at``, ``is_expiring`` would treat it as never-expiring, and
        the token would never refresh again — silent correctness failure.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload()
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(
            _http_response(body={"access_token": "fresh", "expires_in": "3600"}),
        )

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        new_payload = json.loads(crypto_box.derive_account_subkey(account_id).decrypt(new_blob))
        assert "expires_at" in new_payload
        # is_expiring on the new payload should be False (token is fresh).
        assert is_expiring(new_payload) is False

    @pytest.mark.asyncio
    async def test_persists_new_access_token_and_expires_at(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload()
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(
            _http_response(body={"access_token": "fresh-at", "expires_in": 3600})
        )

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        # The UPDATE call carried fresh ciphertext+nonce. Decrypt them and
        # confirm the new token is in the payload.
        conn.execute.assert_awaited_once()
        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        new_payload = json.loads(crypto_box.derive_account_subkey(account_id).decrypt(new_blob))
        assert new_payload["access_token"] == "fresh-at"
        # expires_at is updated to ~1 hour out.
        assert "expires_at" in new_payload

    @pytest.mark.asyncio
    async def test_rotates_refresh_token_when_returned(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload()
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(
            _http_response(body={"access_token": "fresh", "refresh_token": "rt-2"})
        )

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        assert (
            json.loads(crypto_box.derive_account_subkey(account_id).decrypt(new_blob))[
                "refresh_token"
            ]
            == "rt-2"
        )

    @pytest.mark.asyncio
    async def test_keeps_refresh_token_when_omitted(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload()
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        # Response omits refresh_token.
        client = _async_client_returning(_http_response(body={"access_token": "fresh"}))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        assert (
            json.loads(crypto_box.derive_account_subkey(account_id).decrypt(new_blob))[
                "refresh_token"
            ]
            == "rt-1"
        )  # preserved

    @pytest.mark.asyncio
    async def test_http_error_raises_oauth_refresh_error(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload()
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(status=401))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
            pytest.raises(OAuthRefreshError),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

        # Row not updated when refresh fails.
        conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_response_raises(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        payload = _expiring_oauth_payload()
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        # 200 OK but missing access_token in body.
        client = _async_client_returning(_http_response(body={"expires_in": 3600}))

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(httpx, "AsyncClient", MagicMock(return_value=client)),
            pytest.raises(OAuthRefreshError, match="access_token"),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

    @pytest.mark.asyncio
    async def test_no_credential_found_raises(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        conn = _conn_with_transaction()
        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=None),
            ),
            pytest.raises(OAuthRefreshError, match="no active credential"),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )

    @pytest.mark.asyncio
    async def test_missing_refresh_fields_raises(self, crypto_box: CryptoBox) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        # Stored credential is expiring but lacks refresh_token / token_endpoint.
        payload = {
            "access_token": "old",
            "expires_at": _expiring_oauth_payload()["expires_at"],
            "client_id": "cid",
        }
        blob = crypto_box.derive_account_subkey(account_id).encrypt(json.dumps(payload))
        conn = _conn_with_transaction()

        with (
            patch.object(
                queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            pytest.raises(OAuthRefreshError, match="missing required refresh fields"),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                target_url="https://mcp.example.com",
                account_id=account_id,
            )


def test_refresh_skew_seconds_is_positive() -> None:
    """Sanity check the constant — 0 would cause infinite refresh churn."""
    assert REFRESH_SKEW_SECONDS > 0


class TestDeriveAccountSubkey:
    """``CryptoBox.derive_account_subkey`` — HKDF-SHA256 per-account key
    derivation (#367 follow-up). The building block for moving from a
    single deployment-wide vault key to per-account encryption."""

    def _master(self) -> CryptoBox:
        # Use a fixed master so the derive output is deterministic across runs.
        return CryptoBox(b"\xab" * 32)

    def test_returns_a_cryptobox(self) -> None:
        sub = self._master().derive_account_subkey("acc_alpha")
        assert isinstance(sub, CryptoBox)

    def test_subkey_round_trips_plaintext(self) -> None:
        sub = self._master().derive_account_subkey("acc_alpha")
        blob = sub.encrypt("secret payload")
        assert sub.decrypt(blob) == "secret payload"

    def test_deterministic_per_account(self) -> None:
        """Same account_id → same subkey bytes (allowing the same blob
        to decrypt across process restarts)."""
        a1 = self._master().derive_account_subkey("acc_alpha")
        a2 = self._master().derive_account_subkey("acc_alpha")
        blob = a1.encrypt("payload")
        assert a2.decrypt(blob) == "payload"

    def test_subkeys_for_different_accounts_diverge(self) -> None:
        """Two tenants get two unrelated keys — a payload encrypted with
        one cannot be decrypted with the other."""
        from aios.errors import CryptoDecryptError

        a = self._master().derive_account_subkey("acc_alpha")
        b = self._master().derive_account_subkey("acc_beta")
        blob = a.encrypt("payload")
        with pytest.raises(CryptoDecryptError):
            b.decrypt(blob)

    def test_master_cannot_decrypt_subkey_ciphertext(self) -> None:
        """One-way property: a subkey is unreachable from the master
        even though it was derived from it."""
        from aios.errors import CryptoDecryptError

        master = self._master()
        sub = master.derive_account_subkey("acc_alpha")
        blob = sub.encrypt("payload")
        with pytest.raises(CryptoDecryptError):
            master.decrypt(blob)

    def test_empty_account_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="account_id must be non-empty"):
            self._master().derive_account_subkey("")

    def test_golden_vector_pins_derivation_scheme(self) -> None:
        """Byte-for-byte pin of the HKDF construction, exercised through
        the layer that actually encrypts rows: a blob sealed under the
        raw golden subkey bytes must decrypt via ``derive_account_subkey``.
        If this fails, a refactor changed the derivation and every
        existing encrypted row in every deployment just became
        undecryptable — captured before ``derive_subkey_bytes`` was
        factored out of this method."""
        golden = bytes.fromhex("29fa2ed9579b6b8d19b9149f90c16d4618f5e8752494da2fcdae11d934ca2b61")
        blob = CryptoBox(golden).encrypt("sealed-under-golden-bytes")
        subkey = CryptoBox(bytes(range(32))).derive_account_subkey("acct_GOLDEN")
        assert subkey.decrypt(blob) == "sealed-under-golden-bytes"

    def test_derive_subkey_bytes_domain_separation(self) -> None:
        master = self._master()
        assert master.derive_subkey_bytes("ctx-one") != master.derive_subkey_bytes("ctx-two")
        assert master.derive_subkey_bytes("ctx-one") == master.derive_subkey_bytes("ctx-one")


class TestMintSecretPlaceholder:
    """``mint_secret_placeholder`` — the opaque per-(session, credential)
    stand-in (#873). Deterministic by design: it must survive container
    recycles and re-derive identically on any worker sharing the vault
    key, while staying unique per session and unlinkable to the secret."""

    def _salt(self) -> bytes:
        return b"\xcd" * 32

    def test_format(self) -> None:
        placeholder = mint_secret_placeholder(self._salt(), "sess_A", "vcred_1")
        assert placeholder.startswith(SECRET_PLACEHOLDER_PREFIX)
        suffix = placeholder.removeprefix(SECRET_PLACEHOLDER_PREFIX)
        assert len(suffix) == 32
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_deterministic_and_distinct_per_input(self) -> None:
        subkey = self._salt()
        base = mint_secret_placeholder(subkey, "sess_A", "vcred_1")
        assert mint_secret_placeholder(subkey, "sess_A", "vcred_1") == base
        assert mint_secret_placeholder(subkey, "sess_B", "vcred_1") != base
        assert mint_secret_placeholder(subkey, "sess_A", "vcred_2") != base
        other_salt = b"\xee" * 32
        assert mint_secret_placeholder(other_salt, "sess_A", "vcred_1") != base
        assert mint_secret_placeholder(other_salt, "sess_A", "vcred_1") == mint_secret_placeholder(
            other_salt, "sess_A", "vcred_1"
        )


class TestServiceWiringIsAccountScoped:
    """End-to-end integration: a vault_credential blob written by the service
    under ``account_a`` must NOT be decryptable when fetched under ``account_b``.

    The unit-level isolation in :class:`TestDeriveAccountSubkey` proves the
    primitive; this class proves the service code paths actually use the
    primitive. A regression here would mean the wiring was reverted or a
    new call site bypassed the subkey.
    """

    @pytest.mark.asyncio
    async def test_update_credential_cross_account_decrypt_fails(
        self, crypto_box: CryptoBox
    ) -> None:
        # Encrypt as account_a writes (matches create_vault_credential path).
        blob_for_a = crypto_box.derive_account_subkey("acc_a").encrypt(
            json.dumps({"access_token": "secret-of-a"})
        )
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)

        with (
            patch.object(
                queries,
                "get_vault_credential_with_blob",
                AsyncMock(return_value=(_existing_credential(), blob_for_a)),
            ),
            patch.object(
                queries,
                "update_vault_credential",
                AsyncMock(return_value=_existing_credential()),
            ),
            pytest.raises(CryptoDecryptError),
        ):
            # account_b's service call must fail when trying to decrypt
            # account_a's blob — proves the WHERE-clause defense is paired
            # with crypto-layer defense.
            await vaults_service.update_vault_credential(
                pool,
                crypto_box,
                vault_id="vlt_1",
                credential_id="vc_1",
                body=VaultCredentialUpdate(display_name="renamed"),
                account_id="acc_b",
            )

    @pytest.mark.asyncio
    async def test_same_account_decrypt_succeeds(self, crypto_box: CryptoBox) -> None:
        """Positive case — completes the regression net.

        The cross-account test above asserts a *failure* path, which a
        hypothetical revert to ``crypto_box.decrypt_dict`` would *also*
        satisfy (the master key can't decrypt a subkey blob either).
        This test exercises the success path: a blob written under
        account A's subkey must decrypt cleanly when the service is
        called with the same account_id. If the wiring is reverted, the
        decrypt happens with the master key against subkey-encrypted
        bytes and this test fails.
        """
        blob_for_a = crypto_box.derive_account_subkey("acc_a").encrypt(
            json.dumps({"access_token": "secret-of-a"})
        )
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)

        with (
            patch.object(
                queries,
                "get_vault_credential_with_blob",
                AsyncMock(return_value=(_existing_credential(), blob_for_a)),
            ),
            patch.object(
                queries,
                "update_vault_credential",
                AsyncMock(return_value=_existing_credential()),
            ),
        ):
            # Same account_id as the blob was encrypted under — must succeed.
            await vaults_service.update_vault_credential(
                pool,
                crypto_box,
                vault_id="vlt_1",
                credential_id="vc_1",
                body=VaultCredentialUpdate(display_name="renamed"),
                account_id="acc_a",
            )


# ── #879 env-var credential containment gate ─────────────────────────────────


class TestEnvVarCredentialContainment:
    """Pure-function verdict for the two-layer #879 gate
    (:func:`env_var_credential_containment_error`).

    No DB, no async: the caller supplies the per-credential ``allowed_hosts``
    tuples already fetched, so the helper stays unit-testable."""

    def test_no_credentials_returns_none_even_without_env(self) -> None:
        assert env_var_credential_containment_error(None, []) is None

    def test_no_credentials_returns_none_with_unrestricted_env(self) -> None:
        env = EnvironmentConfig(networking=UnrestrictedNetworking())
        assert env_var_credential_containment_error(env, []) is None

    def test_unrestricted_env_with_creds_rejected(self) -> None:
        env = EnvironmentConfig(networking=UnrestrictedNetworking())
        verdict = env_var_credential_containment_error(env, [("api.github.com",)])
        assert verdict is not None
        assert "Limited" in verdict
        # An env IS bound (just not Limited) — the message must say so, distinct
        # from the no-environment-configured branch below.
        assert "networking is not" in verdict

    def test_no_env_config_with_creds_rejected(self) -> None:
        verdict = env_var_credential_containment_error(None, [("api.github.com",)])
        assert verdict is not None
        # No environment bound at all — the message must NOT point the operator
        # at a networking setting that doesn't exist.
        assert "no environment configured" in verdict

    def test_no_networking_config_with_creds_rejected(self) -> None:
        # EnvironmentConfig() leaves networking unset (None) — not Limited.
        verdict = env_var_credential_containment_error(EnvironmentConfig(), [("api.github.com",)])
        assert verdict is not None
        # An env IS bound; networking is just unset — the "not 'limited'" branch.
        assert "networking is not" in verdict

    def test_covered_host_passes(self) -> None:
        env = limited_env("api.github.com")
        assert env_var_credential_containment_error(env, [("api.github.com",)]) is None

    def test_host_comparison_is_case_insensitive(self) -> None:
        # DNS hostnames are case-insensitive and ``HOSTNAME_RE`` permits
        # uppercase, so containment must case-fold both sides. Either casing on
        # either side is covered, in either direction.
        cred_upper_env_lower = limited_env("api.github.com")
        assert (
            env_var_credential_containment_error(cred_upper_env_lower, [("API.GitHub.Com",)])
            is None
        )
        cred_lower_env_upper = limited_env("API.GITHUB.COM")
        assert (
            env_var_credential_containment_error(cred_lower_env_upper, [("api.github.com",)])
            is None
        )

    def test_uncovered_host_rejected_names_offending_host(self) -> None:
        env = limited_env("api.github.com")
        verdict = env_var_credential_containment_error(env, [("evil.example.com",)])
        assert verdict is not None
        assert "'evil.example.com'" in verdict

    def test_cred_path_prefix_compared_host_only(self) -> None:
        # A credential with a path prefix only tightens below an allowed host.
        env = limited_env("api.github.com")
        assert (
            env_var_credential_containment_error(env, [("api.github.com/repos/eumemic",)]) is None
        )

    def test_env_side_parsed_through_grammar_authority(self) -> None:
        # The env side is parsed through ``parse_allowed_host_entry`` too (not a
        # raw string compare), so a bare cred host matches a bare env host. NOTE:
        # ``LimitedNetworking.allowed_hosts`` rejects path-prefixed entries
        # (bare hostnames only — see HOSTNAME_RE), so an env *path prefix* is
        # unconstructable; the host-only parse is what keeps env↔cred comparison
        # symmetric with the path-prefixed CRED side.
        env = limited_env("api.github.com")
        assert env_var_credential_containment_error(env, [("api.github.com",)]) is None

    def test_multiple_creds_one_uncovered_rejected(self) -> None:
        env = limited_env("api.github.com")
        verdict = env_var_credential_containment_error(env, [("api.github.com",), ("pypi.org",)])
        assert verdict is not None
        assert "'pypi.org'" in verdict

    def test_cred_with_empty_allowed_hosts_still_requireslimited_env(self) -> None:
        # An empty allowed_hosts tuple still counts as "has a credential":
        # the OUTER list is non-empty, so the Limited check still fires.
        env = EnvironmentConfig(networking=UnrestrictedNetworking())
        verdict = env_var_credential_containment_error(env, [()])
        assert verdict is not None
        assert "Limited" in verdict

    def test_ip_literal_env_host_is_skipped_not_crash(self) -> None:
        # LimitedNetworking.allowed_hosts accepts IP literals (HOSTNAME_RE), but
        # the stricter credential grammar (parse_allowed_host_entry) rejects
        # them. The env-side extraction must SKIP such entries rather than let
        # the ValueError propagate as a crash. The remaining real env host still
        # covers the cred, so the verdict is None.
        env = limited_env("192.168.1.1", "api.github.com")
        assert env_var_credential_containment_error(env, [("api.github.com",)]) is None

    def test_env_with_only_ip_host_rejects_cred_cleanly(self) -> None:
        # An env whose only allowed_host is an IP literal covers no credential
        # host (IP entries are skipped). The cred is rejected with the actionable
        # message naming its host — a clean rejection, not a ValueError crash.
        env = limited_env("192.168.1.1")
        verdict = env_var_credential_containment_error(env, [("api.github.com",)])
        assert verdict is not None
        assert "'api.github.com'" in verdict

    def test_limited_env_with_empty_cred_hosts_vacuously_passes(self) -> None:
        # A credential with an empty allowed_hosts tuple under a Limited env
        # returns None: the inner host loop never executes, so ∅ ⊆ env is
        # vacuously true. Such a credential is inert — there is no host for the
        # egress swap proxy to DNAT — and is prevented at create by
        # VaultCredentialCreate._validate_shape. Note Check 1 (requires Limited)
        # STILL fires for empty-hosts creds under Unrestricted, which
        # test_cred_with_empty_allowed_hosts_still_requireslimited_env covers.
        assert env_var_credential_containment_error(limited_env("api.github.com"), [()]) is None


def _evc_row(*allowed_hosts: str) -> EnvVarCredentialRow:
    """One ``EnvVarCredentialRow`` carrying ``allowed_hosts`` — the only field
    the attach gate reads. The blob is a throwaway (never decrypted here)."""
    return EnvVarCredentialRow(
        credential_id="vcred_01TEST",
        secret_name="GITHUB_TOKEN",
        allowed_hosts=tuple(allowed_hosts),
        blob=EncryptedBlob(ciphertext=b"x", nonce=b"y"),
        updated_at=datetime(2026, 6, 10, tzinfo=UTC),
    )


class TestEnvVarCredentialAttachGate:
    """The in-transaction advisory gate (:func:`_assert_env_var_creds_contained`)
    that produces a fast 422 at attach. ``queries`` is patched for the two reads
    (credential list + env config), mirroring the call-site test style above."""

    @pytest.mark.asyncio
    async def test_attach_unrestricted_env_with_creds_raises_422(self) -> None:
        conn = MagicMock()
        env = EnvironmentConfig(networking=UnrestrictedNetworking())
        with (
            patch.object(
                queries,
                "list_session_env_var_credentials",
                AsyncMock(return_value=[_evc_row("api.github.com")]),
            ),
            patch.object(
                queries,
                "get_environment_config_for_session",
                AsyncMock(return_value=env),
            ),
            pytest.raises(ValidationError, match="Limited") as exc,
        ):
            await sessions_service._assert_env_var_creds_contained(
                conn, "sess_01TEST", account_id="acct_x"
            )
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_attach_uncovered_host_raises_422_names_host(self) -> None:
        conn = MagicMock()
        with (
            patch.object(
                queries,
                "list_session_env_var_credentials",
                AsyncMock(return_value=[_evc_row("evil.example.com")]),
            ),
            patch.object(
                queries,
                "get_environment_config_for_session",
                AsyncMock(return_value=limited_env("api.github.com")),
            ),
            pytest.raises(ValidationError, match=r"evil\.example\.com"),
        ):
            await sessions_service._assert_env_var_creds_contained(
                conn, "sess_01TEST", account_id="acct_x"
            )

    @pytest.mark.asyncio
    async def test_attach_covered_host_no_raise(self) -> None:
        conn = MagicMock()
        env_getter = AsyncMock(return_value=limited_env("api.github.com"))
        with (
            patch.object(
                queries,
                "list_session_env_var_credentials",
                AsyncMock(return_value=[_evc_row("api.github.com")]),
            ),
            patch.object(queries, "get_environment_config_for_session", env_getter),
        ):
            await sessions_service._assert_env_var_creds_contained(
                conn, "sess_01TEST", account_id="acct_x"
            )
        # The env config WAS read (the gate didn't short-circuit on no creds).
        env_getter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_attach_no_env_var_creds_skips_env_query(self) -> None:
        conn = MagicMock()
        env_getter = AsyncMock(return_value=limited_env("api.github.com"))
        with (
            patch.object(
                queries,
                "list_session_env_var_credentials",
                AsyncMock(return_value=[]),
            ),
            patch.object(queries, "get_environment_config_for_session", env_getter),
        ):
            await sessions_service._assert_env_var_creds_contained(
                conn, "sess_01TEST", account_id="acct_x"
            )
        # No credentials ⇒ the gate returns before reading the env config.
        env_getter.assert_not_awaited()
