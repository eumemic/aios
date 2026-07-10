"""Unit tests for the model_providers models + service layer.

Covers the pure guard predicate (``provider_auth_conflict``) exhaustively —
including the red-team-confirmed bypass fixes (truthiness not presence on the
self-supplied-key exemption; a bare no-row and an unresolvable-model
resolution both route through the same not-root check) and the
code-review-confirmed empty-own-row bypass (an account's own row with an
empty ``api_key`` must not be creatable) — plus the service wrappers:
encryption under the caller/owner subkey, the update omitted-vs-null mapping,
``_resolve_provider_auth``'s provider-derivation short-circuit,
``_check_provider_auth_conflict``'s root-lookup-only-when-needed behavior,
and ``resolve_provider_auth_or_conflict``'s fused resolve+check contract.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.crypto.vault import KEY_BYTES, CryptoBox
from aios.db.queries import ResolvedModelProvider
from aios.db.queries.model_providers import _row_to_model_provider
from aios.errors import ConflictError, CryptoDecryptError
from aios.models.accounts import Account, AccountConfig
from aios.models.model_providers import ProviderAuth, provider_auth_conflict
from aios.services import model_providers as service
from tests.unit.conftest import fake_pool_yielding_conn


@pytest.fixture(autouse=True)
def _unit_provider_auth_ungated() -> None:
    """Shadow (by name) the conftest-level autouse stub of the same name.

    That fixture globally replaces ``resolve_provider_auth_or_conflict`` with
    a clean-pass mock so every OTHER unit test doesn't need to know about the
    guard. THIS file tests the real implementations directly, so it must not
    stub them — a fixture defined in a test module shadows a conftest fixture
    of the same name for that module.
    """
    return None


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(KEY_BYTES))


def _account(account_id: str, *, parent_account_id: str | None) -> Account:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Account(
        id=account_id,
        parent_account_id=parent_account_id,
        can_mint_children=False,
        display_name=account_id,
        metadata={},
        config=AccountConfig(),
        created_at=now,
    )


# ─── provider_auth_conflict: the pure guard predicate ─────────────────────────


class TestProviderAuthConflict:
    def test_no_redirect_never_conflicts(self) -> None:
        assert (
            provider_auth_conflict(
                litellm_extra=None, resolved=None, account_id="acc_x", account_is_root=False
            )
            is False
        )
        assert (
            provider_auth_conflict(
                litellm_extra={}, resolved=None, account_id="acc_x", account_is_root=False
            )
            is False
        )

    def test_self_owned_row_admits_redirect(self) -> None:
        resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_x")
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example"},
                resolved=resolved,
                account_id="acc_x",
                account_is_root=False,
            )
            is False
        )

    def test_ancestor_owned_row_conflicts(self) -> None:
        resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_parent")
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example"},
                resolved=resolved,
                account_id="acc_child",
                account_is_root=False,
            )
            is True
        )

    def test_no_row_root_admits(self) -> None:
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example"},
                resolved=None,
                account_id="acc_root",
                account_is_root=True,
            )
            is False
        )

    def test_no_row_non_root_conflicts(self) -> None:
        """Env keys are root-owned; a non-root account redirecting onto the
        worker's env key is the same exfil shape as an ancestor row."""
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example"},
                resolved=None,
                account_id="acc_child",
                account_is_root=False,
            )
            is True
        )

    def test_unresolvable_model_is_indistinguishable_from_no_row(self) -> None:
        """Fix #2 (red team, severity 90): resolve_provider_auth returns None
        both when there's genuinely no row AND when the model string couldn't
        be resolved to a provider. Both MUST route through the same
        not-root check — there is no third "nothing to check" arm.
        """
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example"},
                resolved=None,  # could be either cause; the predicate can't tell and mustn't try
                account_id="acc_child",
                account_is_root=False,
            )
            is True
        )

    def test_truthy_self_supplied_key_exempts(self) -> None:
        resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_parent")
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example", "api_key": "real-looking-value"},
                resolved=resolved,
                account_id="acc_child",
                account_is_root=False,
            )
            is False
        )

    @pytest.mark.parametrize("falsy_key", [None, ""])
    def test_falsy_self_supplied_key_does_not_exempt(self, falsy_key: str | None) -> None:
        """Fix #1 (red team, severity 100 — CONFIRMED, complete bypass): a
        PRESENT-but-falsy api_key must NOT exempt the guard. litellm treats
        ``api_key=None``/``""`` identically to omitting it (env fallback), so
        a presence-only check lets an attacker who knows no real credential
        fully bypass the guard with a null value.
        """
        resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_parent")
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example", "api_key": falsy_key},
                resolved=resolved,
                account_id="acc_child",
                account_is_root=False,
            )
            is True
        )

    @pytest.mark.parametrize("degenerate_key", ["   ", "\t\n", 12345, {"nested": 1}])
    def test_whitespace_or_nonstr_self_supplied_key_does_not_exempt(
        self, degenerate_key: object
    ) -> None:
        """Code-review hardening: a whitespace-only or non-str api_key must NOT
        exempt the guard. The guard's correctness must not depend on
        un-contracted LiteLLM/httpx handling of a degenerate key (whitespace
        is sent literally and httpx rejects it — but that's implementation
        happenstance, not a contract). The exemption is a stripped-non-empty-
        string check, so these fall through to the ancestor-owned conflict."""
        resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_parent")
        assert (
            provider_auth_conflict(
                litellm_extra={"api_base": "https://x.example", "api_key": degenerate_key},
                resolved=resolved,
                account_id="acc_child",
                account_is_root=False,
            )
            is True
        )

    def test_base_url_alias_triggers_same_as_api_base(self) -> None:
        resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_parent")
        assert (
            provider_auth_conflict(
                litellm_extra={"base_url": "https://x.example"},
                resolved=resolved,
                account_id="acc_child",
                account_is_root=False,
            )
            is True
        )


# ─── wire models: empty api_key rejected ──────────────────────────────────────


class TestEmptyApiKeyRejected:
    """Code-review finding (xhigh pass): api_key had no min_length, so an
    account could create/update its OWN row with an empty key. The guard's
    own-row exemption (provider_auth_conflict) would then admit a redirect
    against that empty-key row, and litellm treats a falsy api_key exactly
    like an omitted one — falling back to the worker's env-owned key. Same
    exfiltration shape as the already-fixed falsy-litellm_extra-api_key
    bypass, reached via a self-created empty row instead.
    """

    def test_create_rejects_empty_api_key(self) -> None:
        from pydantic import ValidationError

        from aios.models.model_providers import ModelProviderCreate

        with pytest.raises(ValidationError):
            ModelProviderCreate(provider="anthropic", api_key="")

    def test_update_rejects_empty_api_key(self) -> None:
        from pydantic import ValidationError

        from aios.models.model_providers import ModelProviderUpdate

        with pytest.raises(ValidationError):
            ModelProviderUpdate(api_key="")

    def test_update_omitted_api_key_still_valid(self) -> None:
        from aios.models.model_providers import ModelProviderUpdate

        assert ModelProviderUpdate().api_key is None


# ─── create/update: encryption + omitted-vs-null mapping ─────────────────────


async def test_create_encrypts_under_caller_subkey(crypto_box: CryptoBox) -> None:
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    captured: dict[str, Any] = {}

    async def _insert(_conn: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _row_to_model_provider(
            {
                "id": "mp_1",
                "provider": kwargs["provider"],
                "api_base": kwargs["api_base"],
                "ciphertext": kwargs["blob"].ciphertext,
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                "updated_at": datetime(2024, 1, 1, tzinfo=UTC),
                "archived_at": None,
            }
        )

    with patch("aios.services.model_providers.queries.insert_model_provider", _insert):
        await service.create_model_provider(
            pool,
            crypto_box,
            account_id="acc_x",
            provider="anthropic",
            api_key="sk-real",
            api_base=None,
        )

    blob = captured["blob"]
    # Round-trips under the CALLER's own subkey.
    assert crypto_box.derive_account_subkey("acc_x").decrypt(blob) == "sk-real"
    # A different account's subkey (or the raw master key) cannot decrypt it.
    with pytest.raises(CryptoDecryptError):
        crypto_box.derive_account_subkey("acc_y").decrypt(blob)


async def test_update_omitted_api_base_passes_ellipsis(crypto_box: CryptoBox) -> None:
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    captured: dict[str, Any] = {}

    async def _update(_conn: Any, _id: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _row_to_model_provider(
            {
                "id": "mp_1",
                "provider": "anthropic",
                "api_base": None,
                "ciphertext": b"",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                "updated_at": datetime(2024, 1, 1, tzinfo=UTC),
                "archived_at": None,
            }
        )

    with patch("aios.services.model_providers.queries.update_model_provider", _update):
        await service.update_model_provider(
            pool, crypto_box, "mp_1", account_id="acc_x", api_key=None
        )

    assert captured["blob"] is None  # api_key=None → keep existing, no re-encrypt
    assert captured["api_base"] is ...  # omitted → Ellipsis sentinel, unchanged


async def test_update_explicit_null_api_base_clears(crypto_box: CryptoBox) -> None:
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    captured: dict[str, Any] = {}

    async def _update(_conn: Any, _id: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _row_to_model_provider(
            {
                "id": "mp_1",
                "provider": "anthropic",
                "api_base": None,
                "ciphertext": b"",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                "updated_at": datetime(2024, 1, 1, tzinfo=UTC),
                "archived_at": None,
            }
        )

    with patch("aios.services.model_providers.queries.update_model_provider", _update):
        await service.update_model_provider(
            pool, crypto_box, "mp_1", account_id="acc_x", api_key=None, api_base=None
        )

    assert captured["api_base"] is None  # explicit null → clear, not Ellipsis


async def test_update_service_layer_uses_presence_not_truthiness(crypto_box: CryptoBox) -> None:
    """Defense-in-depth for a direct service call bypassing the router's
    Pydantic validation (which now rejects an empty api_key at the wire
    boundary): update_model_provider must treat api_key="" as a PRESENT
    (rotate) value per its own documented contract ("api_key=None means
    keep"), not silently collapse it to "keep" via truthiness.
    """
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    captured: dict[str, Any] = {}

    async def _update(_conn: Any, _id: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _row_to_model_provider(
            {
                "id": "mp_1",
                "provider": "anthropic",
                "api_base": None,
                "ciphertext": b"",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                "updated_at": datetime(2024, 1, 1, tzinfo=UTC),
                "archived_at": None,
            }
        )

    with patch("aios.services.model_providers.queries.update_model_provider", _update):
        await service.update_model_provider(
            pool, crypto_box, "mp_1", account_id="acc_x", api_key=""
        )

    assert captured["blob"] is not None  # "" is present, not omitted — must (attempt to) rotate


async def test_update_racing_archive_raises_conflict(crypto_box: CryptoBox) -> None:
    """update_model_provider mirrors update_vault's guarded pattern — a
    concurrent archive winning the race surfaces as ConflictError, not a
    silent no-op or a resurrected secret."""
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)

    with (
        patch(
            "aios.services.model_providers.queries.update_model_provider",
            AsyncMock(side_effect=ConflictError("model provider config mp_1 is archived")),
        ),
        pytest.raises(ConflictError),
    ):
        await service.update_model_provider(
            pool, crypto_box, "mp_1", account_id="acc_x", api_key="sk-new"
        )


# ─── resolve_provider_auth ─────────────────────────────────────────────────────


async def test_resolve_provider_auth_unresolvable_model_skips_db(crypto_box: CryptoBox) -> None:
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not touch the pool"))

    result = await service._resolve_provider_auth(
        pool, crypto_box, account_id="acc_x", model="totally-bogus-model-xyz", litellm_extra=None
    )
    assert result is None


@pytest.mark.parametrize("bad_override", [{"nested": 1}, [1, 2], 123, True])
async def test_resolve_provider_auth_nonstr_custom_llm_provider_does_not_raise(
    crypto_box: CryptoBox, bad_override: object
) -> None:
    """Code-review blocking fix: a truthy non-str custom_llm_provider (script
    params are unvalidated dict[str, Any]) must NOT raise. Before the isinstance
    normalization, an unhashable value hit the lru_cache key machinery and
    raised TypeError BEFORE _derive_provider's own try/except — which on the
    run_llm lane escaped as a silent infinite re-dispatch loop. It's now
    normalized to None (no override), so resolution derives from the model
    string and returns cleanly."""
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=None),
    ):
        result = await service._resolve_provider_auth(
            pool,
            crypto_box,
            account_id="acc_x",
            model="anthropic/claude-x",
            litellm_extra={"custom_llm_provider": bad_override},
        )
    assert result is None  # no raise — derived anthropic, no row


async def test_resolve_provider_auth_decrypts_with_owner_subkey(crypto_box: CryptoBox) -> None:
    owner_subkey = crypto_box.derive_account_subkey("acc_parent")
    blob = owner_subkey.encrypt("sk-ancestor")
    resolved = ResolvedModelProvider(
        owner_account_id="acc_parent", api_base="https://proxy.example", blob=blob
    )
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)

    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=resolved),
    ):
        auth = await service._resolve_provider_auth(
            pool,
            crypto_box,
            account_id="acc_child",
            model="anthropic/claude-x",
            litellm_extra=None,
        )

    assert auth is not None
    assert auth.api_key == "sk-ancestor"
    assert auth.api_base == "https://proxy.example"
    assert auth.owner_account_id == "acc_parent"


async def test_resolve_provider_auth_no_row_returns_none(crypto_box: CryptoBox) -> None:
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=None),
    ):
        auth = await service._resolve_provider_auth(
            pool, crypto_box, account_id="acc_x", model="anthropic/claude-x", litellm_extra=None
        )
    assert auth is None


async def test_resolve_provider_auth_honors_custom_llm_provider_override() -> None:
    """The guard's provider derivation must match what litellm will actually
    dispatch to (fix #2) — passing custom_llm_provider changes which
    provider get_llm_provider resolves to."""
    import litellm

    with_override = litellm.get_llm_provider("gpt-4", custom_llm_provider="anthropic")
    without_override = litellm.get_llm_provider("gpt-4")
    assert with_override[1] != without_override[1]


# ─── check_provider_auth_conflict: root-lookup-only-when-needed ──────────────


async def test_check_conflict_skips_db_when_no_redirect(crypto_box: CryptoBox) -> None:
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not touch the pool"))
    result = await service._check_provider_auth_conflict(
        pool, account_id="acc_x", litellm_extra=None, resolved=None
    )
    assert result is None


async def test_check_conflict_skips_db_when_extra_has_no_redirect_key(
    crypto_box: CryptoBox,
) -> None:
    """A non-empty litellm_extra that carries no api_base/base_url (e.g. just
    sampling knobs) must not trigger the root lookup — gating on truthiness
    of the whole dict rather than on an actual redirect wastes a DB round
    trip on every such call.
    """
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not touch the pool"))
    result = await service._check_provider_auth_conflict(
        pool, account_id="acc_x", litellm_extra={"temperature": 0.7}, resolved=None
    )
    assert result is None


async def test_check_conflict_skips_db_when_row_resolved(crypto_box: CryptoBox) -> None:
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not touch the pool"))
    resolved = ProviderAuth(api_key="k", api_base=None, owner_account_id="acc_x")
    result = await service._check_provider_auth_conflict(
        pool,
        account_id="acc_x",
        litellm_extra={"api_base": "https://x.example"},
        resolved=resolved,
    )
    assert result is None


async def test_check_conflict_looks_up_root_only_on_no_row_arm() -> None:
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    get_account = AsyncMock(return_value=_account("acc_child", parent_account_id="acc_root"))
    with patch("aios.db.queries.get_account", get_account):
        result = await service._check_provider_auth_conflict(
            pool,
            account_id="acc_child",
            litellm_extra={"api_base": "https://evil.example"},
            resolved=None,
        )
    get_account.assert_awaited_once()
    assert result == service.PROVIDER_AUTH_CONFLICT_MESSAGE


async def test_check_conflict_message_is_static_no_account_ids_or_depth() -> None:
    """Fix #3 (red team, severity 45): the conflict message must not leak
    ancestor account ids or tree depth — it's session-visible."""
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)
    with patch(
        "aios.db.queries.get_account",
        AsyncMock(return_value=_account("acc_child", parent_account_id="acc_root")),
    ):
        result = await service._check_provider_auth_conflict(
            pool,
            account_id="acc_child",
            litellm_extra={"api_base": "https://evil.example"},
            resolved=None,
        )
    assert result is not None
    assert "acc_root" not in result
    assert "acc_child" not in result


# ─── resolve_provider_auth_or_conflict: the fused production entry point ──────


async def test_fused_entry_point_returns_resolved_auth_and_no_conflict(
    crypto_box: CryptoBox,
) -> None:
    owner_subkey = crypto_box.derive_account_subkey("acc_x")
    blob = owner_subkey.encrypt("sk-own")
    resolved = ResolvedModelProvider(owner_account_id="acc_x", api_base=None, blob=blob)
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)

    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=resolved),
    ):
        auth, conflict = await service.resolve_provider_auth_or_conflict(
            pool, crypto_box, account_id="acc_x", model="anthropic/claude-x", litellm_extra=None
        )

    assert conflict is None
    assert auth is not None
    assert auth.api_key == "sk-own"


async def test_fused_entry_point_surfaces_conflict_with_resolved_auth(
    crypto_box: CryptoBox,
) -> None:
    """The conflict check must see the SAME resolved value the guard would
    inject — proving the fusion doesn't let resolve and check drift apart."""
    owner_subkey = crypto_box.derive_account_subkey("acc_parent")
    blob = owner_subkey.encrypt("sk-ancestor")
    resolved = ResolvedModelProvider(owner_account_id="acc_parent", api_base=None, blob=blob)
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)

    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=resolved),
    ):
        auth, conflict = await service.resolve_provider_auth_or_conflict(
            pool,
            crypto_box,
            account_id="acc_child",
            model="anthropic/claude-x",
            litellm_extra={"api_base": "https://evil.example"},
        )

    assert conflict == service.PROVIDER_AUTH_CONFLICT_MESSAGE
    assert auth is not None  # still resolved — the CALLER decides not to use it on conflict
    assert auth.owner_account_id == "acc_parent"
