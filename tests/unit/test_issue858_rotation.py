from __future__ import annotations

import base64
from pathlib import Path

import pytest

from aios.crypto.vault import CryptoBox
from aios.services.vaults import SECRET_PLACEHOLDER_PREFIX, mint_secret_placeholder


def test_versioned_blobs_round_trip_and_legacy_decrypts() -> None:
    box = CryptoBox(bytes(range(32)))
    from nacl.secret import SecretBox

    legacy = SecretBox(bytes(range(32))).encrypt(b"legacy", b"x" * 24).ciphertext

    blob = box.encrypt("payload")
    assert blob.ciphertext[:1] == b"\x01"
    assert box.decrypt(blob) == "payload"
    assert box.decrypt(type(blob)(ciphertext=legacy, nonce=b"x" * 24)) == "legacy"


def test_placeholder_is_stable_across_master_key_rotation() -> None:
    salt = b"s" * 32
    base = mint_secret_placeholder(salt, "sess_A", "vcred_1")
    assert base.startswith(SECRET_PLACEHOLDER_PREFIX)
    assert mint_secret_placeholder(salt, "sess_A", "vcred_1") == base
    assert mint_secret_placeholder(salt, "sess_B", "vcred_1") != base
    assert mint_secret_placeholder(salt, "sess_A", "vcred_2") != base
    assert mint_secret_placeholder(b"t" * 32, "sess_A", "vcred_1") != base


def test_egress_ca_key_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from aios.config import Settings

    monkeypatch.delenv("AIOS_EGRESS_CA_KEY", raising=False)
    secrets = tmp_path / "secrets.env"
    secrets.write_text(
        "AIOS_VAULT_KEY="
        + base64.b64encode(b"v" * 32).decode()
        + "\nAIOS_DB_URL=postgresql://x/y\n"
    )
    with pytest.raises(ValidationError, match=r"egress_ca_key|AIOS_EGRESS_CA_KEY"):
        Settings(_env_file=(str(secrets),))
