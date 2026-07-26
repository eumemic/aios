"""Acceptance coverage for eumemic/eumemic-ops#331 (formerly eumemic/aios#1933).

The reported symptom was an intermittent ``401 Bad credentials`` from
api.github.com that "retrying fixes". Two independent mechanisms could put the
LITERAL ``AIOS_SECRET_PLACEHOLDER_…`` string on the wire, and because the
remote then answered 401, a substitution miss was indistinguishable from a
genuinely bad secret:

* **stale placeholder on resume** — placeholders are keyed on CREDENTIAL ID, so
  a rotated/recreated credential leaves a resumed sandbox holding a placeholder
  bound to a dead id;
* **silent passthrough** — the egress proxy forwarded an unexchangeable
  placeholder verbatim instead of refusing it.

These tests pin the two acceptance criteria from the issue.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import (
    ENV_KEYS_LABEL_KEY,
    VAULT_PLACEHOLDER_KEYS_LABEL_KEY,
    Mount,
    SandboxSpec,
)
from aios.sandbox.registry import SandboxRegistry
from aios.services.vaults import (
    SECRET_PLACEHOLDER_PREFIX,
    ResolvedEnvVarCredential,
    mint_secret_placeholder,
)
from tests.helpers.sandbox import FakeBackend

SALT = bytes(range(32))
OWNER = "sess_x"
SECRET_NAME = "GITHUB_TOKEN"
REF = "aios-sbx-default-sess_x:latest"
BASE_IMAGE = "ghcr.io/eumemic/aios-sandbox:latest"

# Credential A is archived and recreated as B with the SAME secret value and
# the SAME allowed_hosts — only the credential id differs. That is exactly the
# rotation shape from the RCA, and it is why the placeholder (a pure function
# of the credential id) changes even though nothing user-visible did.
CRED_A_ID = "vcred_01AAAA"
CRED_B_ID = "vcred_01BBBB"
SECRET_VALUE = "ghp_THE_REAL_SECRET"


def _placeholder(credential_id: str) -> str:
    return mint_secret_placeholder(SALT, OWNER, credential_id)


PH_A = _placeholder(CRED_A_ID)
PH_B = _placeholder(CRED_B_ID)


def _spec(environment: dict[str, str], snapshot_image: str | None = REF) -> SandboxSpec:
    return SandboxSpec(
        session_id=OWNER,
        instance_id="default",
        workspace=Mount(host_path=cast(Any, "/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment=environment,
        labels={},
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image=BASE_IMAGE,
        snapshot_image=snapshot_image,
    )


def test_rotation_mints_a_new_placeholder_for_the_same_secret_name() -> None:
    """The premise of the whole defect: identical secret + hosts, new id ⇒ new
    placeholder. If this ever stopped holding, the resume path below would be
    testing nothing."""
    assert PH_A != PH_B
    assert PH_A.startswith(SECRET_PLACEHOLDER_PREFIX)
    assert PH_B.startswith(SECRET_PLACEHOLDER_PREFIX)
    # Deterministic: re-deriving is idempotent, so re-injection is safe to do
    # unconditionally on every start.
    assert _placeholder(CRED_B_ID) == PH_B


class TestResumeRebindsPlaceholders:
    """Acceptance test 1: credential A works → archive/recreate as B → resume an
    EXISTING snapshot → the env placeholder maps to B, and A's placeholder is
    absent from the resumed env."""

    @pytest.fixture
    def registry(self) -> tuple[SandboxRegistry, FakeBackend]:
        backend = FakeBackend()
        # The snapshot was committed by a provision that injected credential A.
        backend.image_labels_by_ref[REF] = {
            "aios.base_image": BASE_IMAGE,
            VAULT_PLACEHOLDER_KEYS_LABEL_KEY: SECRET_NAME,
        }
        return SandboxRegistry(backend=backend), backend

    @pytest.mark.asyncio
    async def test_resumed_env_carries_b_and_never_a(
        self, registry: tuple[SandboxRegistry, FakeBackend]
    ) -> None:
        reg, _backend = registry
        # The provision that is resuming re-resolved the session's CURRENTLY
        # bound credentials, so the spec already carries B's placeholder.
        resolved = await reg._resolve_snapshot(OWNER, _spec({SECRET_NAME: PH_B}))

        assert resolved.snapshot_image == REF, "a valid snapshot must still resume"
        assert resolved.environment[SECRET_NAME] == PH_B
        # The old placeholder must not survive anywhere in the resumed env —
        # this is the assertion the issue asks for by name.
        assert PH_A not in resolved.environment.values()

    @pytest.mark.asyncio
    async def test_archived_credential_key_is_emptied_not_inherited(
        self, registry: tuple[SandboxRegistry, FakeBackend]
    ) -> None:
        """The residual hole: credential A is archived and NOT replaced.

        Its ``secret_name`` drops out of the current placeholder set, so no
        ``--env`` would override the value baked into the snapshot and the
        resumed container would inherit A's dead placeholder. The key must be
        explicitly emptied instead — and emptied, not merely absent, because
        only an explicit ``--env K=`` overrides a baked ``ENV K=<stale>``.
        """
        reg, _backend = registry
        resolved = await reg._resolve_snapshot(OWNER, _spec({}))

        assert resolved.environment[SECRET_NAME] == ""
        assert PH_A not in resolved.environment.values()

    @pytest.mark.asyncio
    async def test_unrelated_env_is_untouched(
        self, registry: tuple[SandboxRegistry, FakeBackend]
    ) -> None:
        """Only keys the snapshot recorded as VAULT placeholders are neutralized;
        ordinary session/operator env is not collateral damage."""
        reg, _backend = registry
        resolved = await reg._resolve_snapshot(OWNER, _spec({"UNRELATED": "keep-me"}))

        assert resolved.environment["UNRELATED"] == "keep-me"
        assert resolved.environment[SECRET_NAME] == ""

    @pytest.mark.asyncio
    async def test_pre_331_snapshot_with_env_inventory_is_scrubbed(self) -> None:
        """A snapshot committed before this fix carries no placeholder-keys
        label, so the stale-key diff has nothing to diff against. Treating that
        as "nothing to neutralize" would PRESERVE the original vulnerability
        for exactly the population that has it — a pre-#331 snapshot is
        precisely the artifact that may have baked a placeholder for a
        since-archived credential.

        When the snapshot records its env-key inventory (``aios.env_keys``,
        which predates #331), every recorded key the current provision does not
        re-inject is emptied — scrubbing on SHAPE instead of on name.
        """
        backend = FakeBackend()
        backend.image_labels_by_ref[REF] = {
            "aios.base_image": BASE_IMAGE,
            ENV_KEYS_LABEL_KEY: f"{SECRET_NAME},UNRELATED",
        }
        reg = SandboxRegistry(backend=backend)

        # The archived-and-not-replaced case: SECRET_NAME is absent from the
        # current provision's env, so nothing would override the baked value.
        resolved = await reg._resolve_snapshot(OWNER, _spec({}))

        assert resolved.snapshot_image == REF, "a scrubbable snapshot still resumes"
        assert resolved.environment[SECRET_NAME] == "", "the stale baked key must be emptied"
        assert PH_A not in resolved.environment.values()

    @pytest.mark.asyncio
    async def test_pre_331_snapshot_with_env_inventory_keeps_reinjected_keys(self) -> None:
        """The scrub is a superset of the placeholder keys, but a key the
        CURRENT provision re-injects has a legitimate claim to survive (its
        ``--env`` overrides the baked value anyway) and must not be emptied."""
        backend = FakeBackend()
        backend.image_labels_by_ref[REF] = {
            "aios.base_image": BASE_IMAGE,
            ENV_KEYS_LABEL_KEY: f"{SECRET_NAME},UNRELATED",
        }
        reg = SandboxRegistry(backend=backend)

        resolved = await reg._resolve_snapshot(
            OWNER, _spec({SECRET_NAME: PH_B, "UNRELATED": "keep-me"})
        )

        assert resolved.snapshot_image == REF
        assert resolved.environment[SECRET_NAME] == PH_B
        assert resolved.environment["UNRELATED"] == "keep-me"


def test_provision_stamps_the_placeholder_keys_label() -> None:
    """The resume-time neutralization is only as good as the label it reads, so
    pin that provision actually records the injected placeholder key NAMES —
    and records names only, never a placeholder value or a credential id."""
    cred = ResolvedEnvVarCredential(
        credential_id=CRED_B_ID,
        secret_name=SECRET_NAME,
        secret_value=SECRET_VALUE,
        allowed_hosts=("api.github.com",),
        updated_at=datetime(2026, 7, 13, tzinfo=UTC),
        placeholder=PH_B,
    )
    labels = _labels_for(cred)

    assert labels[VAULT_PLACEHOLDER_KEYS_LABEL_KEY] == SECRET_NAME
    serialized = "\x00".join(f"{k}={v}" for k, v in labels.items())
    assert PH_B not in serialized, "a label must never carry a placeholder value"
    assert CRED_B_ID not in serialized, "a label must never carry a credential id"
    assert SECRET_VALUE not in serialized, "a label must never carry the secret"


def _labels_for(cred: ResolvedEnvVarCredential) -> dict[str, str]:
    """Build a spec through the real assembler and return its labels."""
    import asyncio
    from unittest.mock import AsyncMock

    from aios.sandbox.spec import build_spec_from_session
    from tests.helpers.sandbox import limited_env, patch_build_spec_deps

    async def _run() -> dict[str, str]:
        bundle = patch_build_spec_deps(
            env_config=limited_env("api.github.com"),
            env_var_credentials=AsyncMock(return_value=(cred,)),
        )
        import contextlib

        with contextlib.ExitStack() as stack:
            for ctx in bundle:
                stack.enter_context(ctx)
            plan = await build_spec_from_session(OWNER)
        return dict(plan.spec.labels)

    return asyncio.run(_run())


class TestUnexchangeablePlaceholderIsRefused:
    """Acceptance test 2: a request carrying an unexchangeable placeholder
    produces the explicit substitution-failure error and NEVER transmits the
    literal placeholder upstream.

    Boots the REAL proxy (real TLS, real h11 framing) with the mock upstream
    from the proxy suite, so "never transmitted" is proven by an empty upstream
    request log rather than by inspecting a rewritten buffer.
    """

    @pytest.fixture
    def rotated_cred(self) -> ResolvedEnvVarCredential:
        """The post-rotation state: the proxy knows credential B only."""
        return ResolvedEnvVarCredential(
            credential_id=CRED_B_ID,
            secret_name=SECRET_NAME,
            secret_value=SECRET_VALUE,
            allowed_hosts=("api.github.com",),
            updated_at=datetime(2026, 7, 13, tzinfo=UTC),
            placeholder=PH_B,
        )

    @pytest.mark.asyncio
    async def test_dead_placeholder_refused_and_live_one_still_works(
        self, rotated_cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([rotated_cred])

        # A sandbox that resumed WITHOUT the fix still holds A's placeholder.
        # Pre-#331 this rode out to GitHub verbatim and came back 401 "Bad
        # credentials" — an upstream verdict on a credential GitHub never
        # actually received. Now it is refused before any connection opens.
        dead = await _request(proxy, "api.github.com", "/user", headers={"Authorization": PH_A})

        assert dead.status_code == 421
        assert dead.status_code != 401, "must not be conflatable with an upstream auth verdict"
        assert dead.headers["x-aios-egress-error"] == "placeholder_substitution_failed"
        assert "placeholder substitution failed" in dead.text
        assert captured == [], "the literal placeholder must never reach the upstream"

        # ...and the CURRENT placeholder still swaps to the real secret, so the
        # fence refuses only genuine misses. This is the "B authenticates"
        # half of the acceptance criterion.
        live = await _request(proxy, "api.github.com", "/user", headers={"Authorization": PH_B})

        assert live.status_code == 200
        assert len(captured) == 1
        assert captured[0].headers["authorization"] == SECRET_VALUE
        assert PH_B not in str(captured[0].headers)

    @pytest.mark.asyncio
    async def test_dead_placeholder_in_body_is_refused(
        self, rotated_cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """Bodies are swapped too, so they are scanned too — a placeholder is
        just as leaked in a JSON payload as in a header."""
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([rotated_cred])
        res = await _request(
            proxy,
            "api.github.com",
            "/graphql",
            method="POST",
            content=b'{"token": "' + PH_A.encode() + b'"}',
        )

        assert res.status_code == 421
        assert captured == []

    @pytest.mark.asyncio
    async def test_dead_placeholder_inside_basic_auth_is_refused(
        self, rotated_cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """git-over-HTTPS sends the token as a Basic password, where base64
        shifts the byte boundaries — the case a naive substring scan misses,
        and the exact transport in the original report."""
        import base64

        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([rotated_cred])
        blob = base64.b64encode(f"x-access-token:{PH_A}".encode()).decode()
        res = await _request(
            proxy, "api.github.com", "/user", headers={"Authorization": f"Basic {blob}"}
        )

        assert res.status_code == 421
        assert captured == []

    @pytest.mark.asyncio
    async def test_clean_request_without_placeholders_is_unaffected(
        self, rotated_cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """The fence is scoped to the placeholder shape: ordinary traffic
        through the proxy (including a caller's own unrelated bearer token) is
        untouched."""
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([rotated_cred])
        res = await _request(
            proxy, "api.github.com", "/user", headers={"Authorization": "Bearer not-a-placeholder"}
        )

        assert res.status_code == 200
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_placeholder_in_query_string_is_refused(
        self, rotated_cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """The request TARGET is never swapped (it is forwarded verbatim), so a
        placeholder in a query parameter could only ever be transmitted
        LITERALLY — the exact leak the fence exists to stop, one field over.

        Refuse before any upstream connection opens rather than forwarding.
        """
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([rotated_cred])
        res = await _request(proxy, "api.github.com", f"/user?access_token={PH_A}")

        assert res.status_code == 421
        assert res.headers["x-aios-egress-error"] == "placeholder_substitution_failed"
        assert captured == [], "a query-string placeholder must never reach the upstream"

    @pytest.mark.asyncio
    async def test_live_placeholder_in_query_string_is_also_refused(
        self, rotated_cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """Even the CURRENT placeholder is refused in the target: the proxy
        cannot swap it there (URLs are forwarded verbatim), so forwarding would
        transmit a literal placeholder and reaching upstream with it is the bug.
        Refusal is the honest outcome — and it names the fix in the body."""
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([rotated_cred])
        res = await _request(proxy, "api.github.com", f"/user?access_token={PH_B}")

        assert res.status_code == 421
        assert captured == []
        assert "QUERY" in res.text, "the refusal must tell the caller why a URL token cannot work"

    @pytest.mark.asyncio
    async def test_refusal_log_never_carries_request_derived_bytes(
        self,
        rotated_cred: ResolvedEnvVarCredential,
        make_proxy: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """THE CRITICAL PROPERTY: the residual scan runs POST-swap, so its input
        contains REAL SECRETS.

        A vault secret is arbitrary operator-supplied text — nothing stops one
        from having the placeholder shape. Here the live credential's REAL
        VALUE is placeholder-shaped, so a SUCCESSFUL substitution looks residual
        to the scanner and the request is refused (fail-closed, correct). The
        thing that must never happen is the refusal log writing the real
        credential — so assert no request-derived bytes reach the logs at all.
        """
        import logging

        shaped_secret = SECRET_PLACEHOLDER_PREFIX + "f" * 32
        cred = ResolvedEnvVarCredential(
            credential_id=CRED_B_ID,
            secret_name=SECRET_NAME,
            secret_value=shaped_secret,
            allowed_hosts=("api.github.com",),
            updated_at=datetime(2026, 7, 13, tzinfo=UTC),
            placeholder=PH_B,
        )
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([cred])
        with caplog.at_level(logging.WARNING):
            res = await _request(proxy, "api.github.com", "/user", headers={"Authorization": PH_B})

        assert res.status_code == 421, "an indistinguishable residual must fail closed"
        assert captured == []
        logged = "\n".join(
            [record.getMessage() for record in caplog.records]
            + [str(record.__dict__) for record in caplog.records]
        )
        assert "placeholder_substitution_failed" in logged, "the refusal must still be visible"
        # The real secret — and every prefix of it long enough to be useful —
        # must be absent. A prefix of a secret is still secret material.
        assert shaped_secret not in logged
        assert shaped_secret[: len(SECRET_PLACEHOLDER_PREFIX) + 8] not in logged
        # The region label is a constant from a closed vocabulary, so it is the
        # one scan-derived thing that may appear.
        assert "authorization_header" in logged


class TestPlaceholderShapeIsBounded:
    """Finding 5: the fence keys on the MINTED placeholder shape, anchored, so
    traffic that merely CONTAINS a placeholder-shaped substring inside a longer
    opaque token is not collateral damage."""

    @pytest.fixture
    def cred(self) -> ResolvedEnvVarCredential:
        return ResolvedEnvVarCredential(
            credential_id=CRED_B_ID,
            secret_name=SECRET_NAME,
            secret_value=SECRET_VALUE,
            allowed_hosts=("api.github.com",),
            updated_at=datetime(2026, 7, 13, tzinfo=UTC),
            placeholder=PH_B,
        )

    @pytest.mark.asyncio
    async def test_longer_token_embedding_the_shape_is_not_refused(
        self, cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """A 33rd hex char makes it a DIFFERENT token, not a placeholder. An
        unanchored match would find a 32-char window inside it and refuse
        legitimate traffic; a real minted placeholder is always delimited."""
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([cred])
        res = await _request(
            proxy,
            "api.github.com",
            "/user",
            headers={"Authorization": f"Bearer {PH_A}deadbeef"},
        )

        assert res.status_code == 200, "a longer opaque token must not be refused"
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_placeholder_shaped_substring_in_a_body_is_not_refused(
        self, cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """Code samples and user data may legitimately mention the prefix (docs,
        a diff, a log line being uploaded). Only a whole, delimited placeholder
        is a substitution miss."""
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([cred])
        body = b"see AIOS_SECRET_PLACEHOLDER_notactuallyhex and " + PH_A.encode() + b"0123"
        res = await _request(proxy, "api.github.com", "/graphql", method="POST", content=body)

        assert res.status_code == 200
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_legitimate_delimited_placeholder_shaped_payload_is_refused_by_design(
        self, cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """THE DELIBERATE TRADE (the MEDIUM finding), pinned as a trade.

        This is a FALSE POSITIVE and we are keeping it. The body below is
        innocent payload — a log line being shipped to an API that happens to
        quote a whole, delimited placeholder — and it is refused with 421.

        Why refusing is correct rather than merely convenient:

        * The scan runs POST-swap, so provenance is gone. An unexchangeable
          placeholder and a placeholder-shaped literal the sandbox typed are
          the same bytes in the same position; nothing left in the request
          separates them. Any narrowing (allow it in bodies but not headers,
          key on content type, ...) would be a GUESS about intent, and a fence
          that guesses is a fence an unlucky miss walks through.
        * The errors are not symmetric. This refusal costs a loud, local 421
          whose body names the cause and the workaround. The other direction
          puts a credential-shaped token on the wire and draws a 401 that
          cannot be told apart from a bad secret — the silent misdiagnosis
          eumemic/eumemic-ops#331 exists to end.

        So the assertions here are about the COST BEING PAID WELL, not about
        the refusal being desirable: the caller must be told exactly what
        happened, and a workaround must exist.
        """
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([cred])
        legitimate_payload = b'{"log": "worker booted with TOKEN=' + PH_A.encode() + b' (masked)"}'
        res = await _request(
            proxy, "api.github.com", "/graphql", method="POST", content=legitimate_payload
        )

        # Fail closed on ambiguity: not sent, even though it was innocent.
        assert res.status_code == 421
        assert captured == [], "the ambiguous request must not reach the upstream"
        # The cost is only acceptable because it is SELF-EXPLAINING: the caller
        # must learn that the shape is what tripped it and that encoding is the
        # way out, without reading our source.
        assert res.headers["x-aios-egress-error"] == "placeholder_substitution_failed"
        assert "SHAPE" in res.text, "the refusal must say it keys on the shape"
        assert "deliberate" in res.text, "...and that the false positive is intended"
        assert "base64" in res.text, "...and name the workaround"

    @pytest.mark.asyncio
    async def test_the_documented_workaround_actually_works(
        self, cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """The trade is only defensible if the refusal body's advice is true.

        The 421 tells the caller to encode or redact placeholder-shaped
        payload. Assert that a caller who does so gets through — otherwise the
        false positive is an unfixable dead end rather than a cost with a
        remedy.
        """
        import base64

        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([cred])
        encoded = base64.b64encode(PH_A.encode())
        res = await _request(
            proxy,
            "api.github.com",
            "/graphql",
            method="POST",
            content=b'{"log": "' + encoded + b'"}',
        )

        assert res.status_code == 200, "the workaround the refusal advertises must work"
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_a_delimited_placeholder_in_prose_is_still_refused(
        self, cred: ResolvedEnvVarCredential, make_proxy: Any
    ) -> None:
        """The boundary relaxation is bounded: a WHOLE placeholder surrounded by
        ordinary delimiters (quotes, spaces, JSON punctuation) is still refused
        — that is how a genuine substitution miss appears in every real
        transport, so relaxing here would reopen the bug."""
        from tests.unit.sandbox.test_secret_egress_proxy import _request

        proxy, captured = await make_proxy([cred])
        res = await _request(
            proxy,
            "api.github.com",
            "/graphql",
            method="POST",
            content=b'{"token": "' + PH_A.encode() + b'"}',
        )

        assert res.status_code == 421
        assert captured == []
