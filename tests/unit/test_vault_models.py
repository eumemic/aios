"""Unit tests for vault Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

from aios.models.vaults import (
    RESERVED_SANDBOX_ENV_KEYS,
    TokenEndpointAuth,
    TokenEndpointAuthBasic,
    TokenEndpointAuthNone,
    TokenEndpointAuthPost,
    VaultCreate,
    VaultCredentialCreate,
    VaultCredentialUpdate,
    VaultUpdate,
    parse_allowed_host_entry,
)


class TestVaultCreate:
    def test_valid_create(self) -> None:
        v = VaultCreate(display_name="my vault")
        assert v.display_name == "my vault"
        assert v.metadata == {}

    def test_with_metadata(self) -> None:
        v = VaultCreate(display_name="v", metadata={"team": "infra"})
        assert v.metadata == {"team": "infra"}

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            VaultCreate(display_name="")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            VaultCreate(display_name="v", bogus="x")  # type: ignore[call-arg]


class TestVaultUpdate:
    def test_all_optional(self) -> None:
        u = VaultUpdate()
        assert u.display_name is None
        assert u.metadata is None


class TestTokenEndpointAuth:
    def test_none_variant_no_secret(self) -> None:
        v = TokenEndpointAuthNone(method="none")
        assert v.method == "none"

    def test_none_variant_rejects_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            TokenEndpointAuthNone(method="none", client_secret=SecretStr("x"))  # type: ignore[call-arg]

    def test_basic_variant_carries_secret(self) -> None:
        v = TokenEndpointAuthBasic(
            method="client_secret_basic",
            client_secret=SecretStr("shh"),
        )
        assert v.method == "client_secret_basic"
        assert v.client_secret.get_secret_value() == "shh"

    def test_post_variant_carries_secret(self) -> None:
        v = TokenEndpointAuthPost(
            method="client_secret_post",
            client_secret=SecretStr("shh"),
        )
        assert v.method == "client_secret_post"
        assert v.client_secret.get_secret_value() == "shh"

    def test_basic_requires_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            TokenEndpointAuthBasic(method="client_secret_basic")  # type: ignore[call-arg]

    def test_post_requires_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            TokenEndpointAuthPost(method="client_secret_post")  # type: ignore[call-arg]

    def test_discriminator_dispatches_none(self) -> None:
        adapter: TypeAdapter[TokenEndpointAuth] = TypeAdapter(TokenEndpointAuth)
        v = adapter.validate_python({"method": "none"})
        assert isinstance(v, TokenEndpointAuthNone)

    def test_discriminator_dispatches_basic(self) -> None:
        adapter: TypeAdapter[TokenEndpointAuth] = TypeAdapter(TokenEndpointAuth)
        v = adapter.validate_python({"method": "client_secret_basic", "client_secret": "shh"})
        assert isinstance(v, TokenEndpointAuthBasic)
        assert v.client_secret.get_secret_value() == "shh"

    def test_discriminator_dispatches_post(self) -> None:
        adapter: TypeAdapter[TokenEndpointAuth] = TypeAdapter(TokenEndpointAuth)
        v = adapter.validate_python({"method": "client_secret_post", "client_secret": "shh"})
        assert isinstance(v, TokenEndpointAuthPost)

    def test_discriminator_rejects_unknown_method(self) -> None:
        adapter: TypeAdapter[TokenEndpointAuth] = TypeAdapter(TokenEndpointAuth)
        with pytest.raises(ValidationError):
            adapter.validate_python({"method": "bogus"})

    def test_secret_masked_in_repr(self) -> None:
        v = TokenEndpointAuthPost(
            method="client_secret_post",
            client_secret=SecretStr("super-secret"),
        )
        assert "super-secret" not in repr(v)


class TestTargetUrlSafety:
    def test_rejects_loopback_target(self) -> None:
        with pytest.raises(ValidationError, match="private or runtime-local host"):
            VaultCredentialCreate(
                auth_type="bearer",
                target_url="http://127.0.0.1:8080/mcp",
                token="secret",
            )

    def test_rejects_runtime_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_URL", "https://runtime.example/v1")
        with pytest.raises(ValidationError, match="private or runtime-local host"):
            VaultCredentialCreate(
                auth_type="bearer",
                target_url="https://runtime.example/mcp",
                token="secret",
            )


class TestVaultCredentialCreate:
    def test_bearer_header_valid(self) -> None:
        c = VaultCredentialCreate(
            target_url="https://mcp.example.com",
            auth_type="bearer_header",
            token=SecretStr("my-token"),
        )
        assert c.auth_type == "bearer_header"
        assert c.token is not None
        assert c.token.get_secret_value() == "my-token"

    def test_oauth2_refresh_valid(self) -> None:
        c = VaultCredentialCreate(
            target_url="https://mcp.example.com",
            auth_type="oauth2_refresh",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
        )
        assert c.auth_type == "oauth2_refresh"
        assert c.access_token is not None

    def test_oauth2_refresh_with_typed_token_endpoint_auth_basic(self) -> None:
        c = VaultCredentialCreate(
            target_url="https://mcp.example.com",
            auth_type="oauth2_refresh",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
            token_endpoint="https://issuer.example/token",
            token_endpoint_auth=TokenEndpointAuthBasic(
                method="client_secret_basic",
                client_secret=SecretStr("shh"),
            ),
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthBasic)
        assert c.token_endpoint_auth.client_secret.get_secret_value() == "shh"

    def test_oauth2_refresh_with_typed_token_endpoint_auth_post(self) -> None:
        c = VaultCredentialCreate(
            target_url="https://mcp.example.com",
            auth_type="oauth2_refresh",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
            token_endpoint="https://issuer.example/token",
            token_endpoint_auth=TokenEndpointAuthPost(
                method="client_secret_post",
                client_secret=SecretStr("shh"),
            ),
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthPost)

    def test_oauth2_refresh_with_typed_token_endpoint_auth_none(self) -> None:
        c = VaultCredentialCreate(
            target_url="https://mcp.example.com",
            auth_type="oauth2_refresh",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
            token_endpoint="https://issuer.example/token",
            token_endpoint_auth=TokenEndpointAuthNone(method="none"),
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthNone)

    def test_token_endpoint_auth_accepts_dict_form(self) -> None:
        c = VaultCredentialCreate.model_validate(
            {
                "target_url": "https://mcp.example.com",
                "auth_type": "oauth2_refresh",
                "access_token": "tok",
                "client_id": "cid",
                "token_endpoint": "https://issuer.example/token",
                "token_endpoint_auth": {
                    "method": "client_secret_basic",
                    "client_secret": "shh",
                },
            }
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthBasic)

    def test_rejects_flat_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                target_url="https://mcp.example.com",
                auth_type="oauth2_refresh",
                access_token=SecretStr("access-tok"),
                client_id="client-123",
                client_secret=SecretStr("flat"),  # type: ignore[call-arg]
            )

    def test_rejects_bad_auth_type(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                target_url="https://x.com",
                auth_type="unknown",
                token=SecretStr("t"),
            )

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                target_url="",
                auth_type="bearer_header",
                token=SecretStr("t"),
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                target_url="https://x.com",
                auth_type="bearer_header",
                token=SecretStr("t"),
                bogus="x",  # type: ignore[call-arg]
            )


class TestEnvironmentVariableCredential:
    def _create(self, **overrides: object) -> VaultCredentialCreate:
        kwargs: dict[str, object] = {
            "auth_type": "environment_variable",
            "secret_name": "GITHUB_TOKEN",
            "allowed_hosts": ["api.github.com"],
            "secret_value": SecretStr("ghp_xxx"),
        }
        kwargs.update(overrides)
        return VaultCredentialCreate(**kwargs)

    def test_valid_bare_host(self) -> None:
        c = self._create()
        assert c.auth_type == "environment_variable"
        assert c.secret_name == "GITHUB_TOKEN"
        assert c.allowed_hosts == ["api.github.com"]
        assert c.target_url is None
        assert c.secret_value is not None
        assert c.secret_value.get_secret_value() == "ghp_xxx"

    def test_valid_host_with_path_prefix(self) -> None:
        c = self._create(allowed_hosts=["api.github.com/repos/eumemic"])
        assert c.allowed_hosts == ["api.github.com/repos/eumemic"]

    def test_explicit_whole_host_canonicalizes_to_bare(self) -> None:
        c = self._create(allowed_hosts=["api.github.com/"])
        assert c.allowed_hosts == ["api.github.com"]

    def test_trailing_slash_prefix_canonicalizes(self) -> None:
        c = self._create(allowed_hosts=["api.github.com/repos/eumemic/"])
        assert c.allowed_hosts == ["api.github.com/repos/eumemic"]

    def test_multiple_hosts(self) -> None:
        c = self._create(allowed_hosts=["api.github.com", "api.tavily.com"])
        assert c.allowed_hosts == ["api.github.com", "api.tavily.com"]

    def test_rejects_missing_secret_name(self) -> None:
        with pytest.raises(ValidationError):
            self._create(secret_name=None)

    def test_rejects_target_url_on_env_var(self) -> None:
        with pytest.raises(ValidationError):
            self._create(target_url="https://x.com")

    def test_rejects_empty_allowed_hosts(self) -> None:
        with pytest.raises(ValidationError):
            self._create(allowed_hosts=[])

    @pytest.mark.parametrize("name", ["1FOO", "FOO-BAR", "FOO BAR", "FOO.BAR", "", "FOO$"])
    def test_rejects_bad_secret_name(self, name: str) -> None:
        with pytest.raises(ValidationError):
            self._create(secret_name=name)

    @pytest.mark.parametrize("name", sorted(RESERVED_SANDBOX_ENV_KEYS))
    def test_rejects_reserved_secret_name(self, name: str) -> None:
        with pytest.raises(ValidationError):
            self._create(secret_name=name)

    def test_rejects_secret_name_too_long(self) -> None:
        with pytest.raises(ValidationError):
            self._create(secret_name="A" * 129)

    @pytest.mark.parametrize(
        "host",
        [
            "api.example.com:443",  # port
            "*.example.com",  # wildcard
            "exa_mple.com",  # underscore
            "127.0.0.1",  # IPv4 literal
            "169.254.169.254",  # metadata IPv4
            "2130706433",  # integer IP form / all-numeric label
            "10",  # bare numeric label
            "::1",  # IPv6
            "0x7f000001",  # hex IPv4 literal (alpha letters, still an IP)
            "0xA.0xB.0xC.0xD",  # dotted hex IPv4 literal
            "0Xa",  # short hex IPv4 literal, uppercase prefix
            "0177.0.0.1",  # octal-leading IPv4 literal
            "api.github.com\n",  # trailing newline — the regex `$` anchor must not admit it
            "api.github.com\nevil.com",  # embedded newline
        ],
    )
    def test_rejects_bad_host(self, host: str) -> None:
        with pytest.raises(ValidationError):
            self._create(allowed_hosts=[host])

    def test_dedups_canonical_allowed_hosts(self) -> None:
        c = self._create(
            allowed_hosts=["api.github.com", "api.github.com/", "api.github.com", "api.tavily.com"]
        )
        # All three github spellings canonicalize to one entry; order preserved.
        assert c.allowed_hosts == ["api.github.com", "api.tavily.com"]

    @pytest.mark.parametrize(
        "host",
        [
            "0xdeadbeef.example.com",  # hex-looking label is fine when it isn't the final label
            "c0ffee.io",  # alphanumeric label with digits
            "3m.com",  # leading-digit label
            "xn--p1ai",  # punycode
            "example.9-9",  # letterless final label that is NOT an IP literal — only IP
            "foo.1-2",  # literals are rejected, not every digits-only-looking label
        ],
    )
    def test_accepts_non_ip_numeric_looking_labels(self, host: str) -> None:
        c = self._create(allowed_hosts=[host])
        assert c.allowed_hosts == [host]

    @pytest.mark.parametrize(
        "entry",
        [
            "api.github.com/repos/../gists",  # dot-segment
            "api.github.com/repos/./x",  # single-dot segment
            "api.github.com//repos",  # empty segment
            "api.github.com/repos/%2e%2e",  # percent-encoding
        ],
    )
    def test_rejects_bad_path_prefix(self, entry: str) -> None:
        with pytest.raises(ValidationError):
            self._create(allowed_hosts=[entry])

    def test_other_secret_fields_ignored_not_rejected(self) -> None:
        # Cross-kind secret fields follow the shared-base accept-and-ignore
        # convention (a bearer create silently ignores username, etc.); only
        # the structural fields are governed by the shape validator.
        c = self._create(token=SecretStr("ignored"))
        assert c.auth_type == "environment_variable"


class TestHeaderCredentialShape:
    @pytest.mark.parametrize("extra", [{"secret_name": "FOO"}, {"allowed_hosts": ["x.com"]}])
    def test_rejects_env_var_fields_on_bearer(self, extra: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                target_url="https://x.com",
                auth_type="bearer_header",
                token=SecretStr("t"),
                **extra,
            )


class TestParseAllowedHostEntry:
    @pytest.mark.parametrize(
        "entry,expected",
        [
            ("api.github.com", ("api.github.com", None)),
            ("api.github.com/", ("api.github.com", None)),
            ("api.github.com/repos/eumemic", ("api.github.com", "/repos/eumemic")),
            ("api.github.com/repos/eumemic/", ("api.github.com", "/repos/eumemic")),
            ("registry.npmjs.org/@myorg", ("registry.npmjs.org", "/@myorg")),
            ("host.example/v1/r:action", ("host.example", "/v1/r:action")),
        ],
    )
    def test_parses(self, entry: str, expected: tuple[str, str | None]) -> None:
        assert parse_allowed_host_entry(entry) == expected

    @pytest.mark.parametrize(
        "entry",
        [
            "",
            "127.0.0.1",
            "0x7f000001",
            "host//x",
            "host/../x",
            "host/%2e",
            "host\n",  # trailing newline in the host part
            "host/repos\n",  # trailing newline in a path segment
        ],
    )
    def test_rejects(self, entry: str) -> None:
        with pytest.raises(ValueError):
            parse_allowed_host_entry(entry)


class TestVaultCredentialUpdate:
    def test_all_optional(self) -> None:
        u = VaultCredentialUpdate()
        assert u.display_name is None
        assert u.token is None
        assert u.access_token is None

    def test_partial_update(self) -> None:
        u = VaultCredentialUpdate(token=SecretStr("new-token"))
        assert u.token is not None
        assert u.token.get_secret_value() == "new-token"
        assert "token" in u.model_fields_set
        assert "access_token" not in u.model_fields_set

    def test_partial_update_token_endpoint_auth(self) -> None:
        u = VaultCredentialUpdate(
            token_endpoint_auth=TokenEndpointAuthPost(
                method="client_secret_post",
                client_secret=SecretStr("rotated"),
            ),
        )
        assert "token_endpoint_auth" in u.model_fields_set
        assert isinstance(u.token_endpoint_auth, TokenEndpointAuthPost)

    def test_rejects_flat_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialUpdate(client_secret=SecretStr("flat"))  # type: ignore[call-arg]
