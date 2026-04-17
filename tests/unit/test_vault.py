"""Tests for the libsodium CryptoBox and vault service-layer helpers."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from aios.crypto.vault import KEY_BYTES, NONCE_BYTES, CryptoBox, EncryptedBlob
from aios.errors import CryptoDecryptError, OAuthRefreshError
from aios.models.vaults import (
    TokenEndpointAuthBasic,
    TokenEndpointAuthNone,
    TokenEndpointAuthPost,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
)
from aios.services import vaults as vaults_service
from aios.services.vaults import (
    REFRESH_SKEW_SECONDS,
    _extract_auth_payload,
    _merge_auth_payload,
    is_expiring,
    refresh_credential,
)
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


def _existing_credential() -> VaultCredential:

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
        pool = fake_pool_yielding_conn(conn)

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
        pool = fake_pool_yielding_conn(conn)

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

        payload = _expiring_oauth_payload(
            expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        )
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        client.post.assert_not_awaited()
        conn.execute.assert_not_awaited()  # row not updated

    @pytest.mark.asyncio
    async def test_basic_method_uses_basic_auth(self, crypto_box: CryptoBox) -> None:
        import httpx as _httpx

        payload = _expiring_oauth_payload(
            token_endpoint_auth={"method": "client_secret_basic", "client_secret": "shh"},
        )
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        kwargs = client.post.await_args.kwargs
        assert isinstance(kwargs["auth"], _httpx.BasicAuth)
        # client_secret never leaks into the form body.
        assert "client_secret" not in kwargs["data"]
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "rt-1"

    @pytest.mark.asyncio
    async def test_post_method_includes_secret_in_body(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload(
            token_endpoint_auth={"method": "client_secret_post", "client_secret": "shh"},
        )
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        kwargs = client.post.await_args.kwargs
        assert "auth" not in kwargs
        assert kwargs["data"]["client_secret"] == "shh"
        assert kwargs["data"]["client_id"] == "cid"

    @pytest.mark.asyncio
    async def test_none_method_includes_only_client_id(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload(
            token_endpoint_auth={"method": "none"},
        )
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(body={"access_token": "new"}))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
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
        payload = _expiring_oauth_payload()
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(
            _http_response(body={"access_token": "fresh", "expires_in": "3600"}),
        )

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        new_payload = json.loads(crypto_box.decrypt(new_blob))
        assert "expires_at" in new_payload
        # is_expiring on the new payload should be False (token is fresh).
        assert is_expiring(new_payload) is False

    @pytest.mark.asyncio
    async def test_persists_new_access_token_and_expires_at(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload()
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(
            _http_response(body={"access_token": "fresh-at", "expires_in": 3600})
        )

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        # The UPDATE call carried fresh ciphertext+nonce. Decrypt them and
        # confirm the new token is in the payload.
        conn.execute.assert_awaited_once()
        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        new_payload = json.loads(crypto_box.decrypt(new_blob))
        assert new_payload["access_token"] == "fresh-at"
        # expires_at is updated to ~1 hour out.
        assert "expires_at" in new_payload

    @pytest.mark.asyncio
    async def test_rotates_refresh_token_when_returned(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload()
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(
            _http_response(body={"access_token": "fresh", "refresh_token": "rt-2"})
        )

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        assert json.loads(crypto_box.decrypt(new_blob))["refresh_token"] == "rt-2"

    @pytest.mark.asyncio
    async def test_keeps_refresh_token_when_omitted(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload()
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        # Response omits refresh_token.
        client = _async_client_returning(_http_response(body={"access_token": "fresh"}))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        args = conn.execute.await_args.args
        new_blob = EncryptedBlob(ciphertext=args[1], nonce=args[2])
        assert json.loads(crypto_box.decrypt(new_blob))["refresh_token"] == "rt-1"  # preserved

    @pytest.mark.asyncio
    async def test_http_error_raises_oauth_refresh_error(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload()
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        client = _async_client_returning(_http_response(status=401))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
            pytest.raises(OAuthRefreshError),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

        # Row not updated when refresh fails.
        conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_response_raises(self, crypto_box: CryptoBox) -> None:
        payload = _expiring_oauth_payload()
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()
        # 200 OK but missing access_token in body.
        client = _async_client_returning(_http_response(body={"expires_in": 3600}))

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            patch.object(vaults_service.httpx, "AsyncClient", MagicMock(return_value=client)),
            pytest.raises(OAuthRefreshError, match="access_token"),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

    @pytest.mark.asyncio
    async def test_no_credential_found_raises(self, crypto_box: CryptoBox) -> None:
        conn = _conn_with_transaction()
        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=None),
            ),
            pytest.raises(OAuthRefreshError, match="no active credential"),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )

    @pytest.mark.asyncio
    async def test_missing_refresh_fields_raises(self, crypto_box: CryptoBox) -> None:
        # Stored credential is expiring but lacks refresh_token / token_endpoint.
        payload = {
            "access_token": "old",
            "expires_at": _expiring_oauth_payload()["expires_at"],
            "client_id": "cid",
        }
        blob = crypto_box.encrypt(json.dumps(payload))
        conn = _conn_with_transaction()

        with (
            patch.object(
                vaults_service.queries,
                "lock_oauth_credential_for_refresh",
                AsyncMock(return_value=("vc_1", blob)),
            ),
            pytest.raises(OAuthRefreshError, match="missing required refresh fields"),
        ):
            await refresh_credential(
                crypto_box,
                conn,
                vault_id="vlt_1",
                mcp_server_url="https://mcp.example.com",
            )


def test_refresh_skew_seconds_is_positive() -> None:
    """Sanity check the constant — 0 would cause infinite refresh churn."""
    assert REFRESH_SKEW_SECONDS > 0
