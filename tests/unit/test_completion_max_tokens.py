"""Outbound ``max_tokens`` defaulting for Anthropic-shaped routes (issue #2451).

**The defect.** aios never set ``max_tokens``. Omitting it does NOT mean
"unlimited" — on an Anthropic-shaped route it means whatever that route's
default is, and for several of them that default is 4096. With extended
thinking enabled, thinking tokens are drawn from that same budget, so a hard
turn can spend the whole 4096 reasoning and emit EMPTY assistant content with
``finish_reason: "length"`` — billed in full, recorded as a clean turn, no
error raised. The failure gets *more* likely the harder the task, which is
exactly inverted from where reliability is wanted.

**What each route actually put on the wire before the fix** (captured from the
real serialized request body through ``call_litellm`` against litellm 1.96.2,
the pinned version):

===============================  ==========================================
route                            ``max_tokens`` sent when the caller omits it
===============================  ==========================================
``anthropic/<mapped model>``     the model's max_output_tokens (litellm's
                                 ``AnthropicConfig.get_config`` fills it in)
``anthropic/<unmapped model>``   4096 (``DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS``)
``vertex_ai/claude-*``           4096
``openrouter/anthropic/*``       absent → the provider's own default
``bedrock/anthropic.*``          absent → the provider's own default
===============================  ==========================================

OpenRouter is deliberately excluded because it validates affordability against
``max_tokens`` and can reject a full-ceiling reservation with HTTP 402. Direct
Anthropic and other Anthropic-shaped routes still receive the explicit ceiling.
The exclusion is keyed on ``custom_llm_provider`` as well as the model prefix,
because LiteLLM dispatches on the override when one is present.

Note which route each test runs on: the precedence rules (a caller value wins)
are only observable on a route the default would otherwise fire on, so they are
pinned against ``anthropic/*``. Asserting them on ``openrouter/*`` would pass
vacuously — the OpenRouter early return alone satisfies them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import litellm
import pytest

from aios.harness import completion


class _DictResponse(dict[str, object]):
    """Subscriptable + ``.get`` + ``_hidden_params``, like a real litellm response."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._hidden_params: dict[str, object] = {}


def _ok_response() -> _DictResponse:
    return _DictResponse(
        choices=[{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        usage={},
    )


async def _capture_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run ``call_litellm`` and return the kwargs handed to ``litellm.acompletion``."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _DictResponse:
        captured.update(kwargs)
        return _ok_response()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    await completion.call_litellm(
        completion.LlmRequest(
            messages=[{"role": "user", "content": "hi"}],
            params=params,
            session_id="sess_max_tokens",
        ),
        model=model,
    )
    return captured


# Paired provider routes pin the deliberate difference in defaulting behavior.
_PROXY_CLAUDE_MODEL = "openrouter/anthropic/claude-opus-4-1"
_DIRECT_CLAUDE_MODEL = "anthropic/claude-opus-4-1"


class TestDefaultMaxTokens:
    @pytest.mark.asyncio
    async def test_openrouter_thinking_request_leaves_max_tokens_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = await _capture_kwargs(
            monkeypatch,
            model=_PROXY_CLAUDE_MODEL,
            params={"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        )

        assert "max_tokens" not in captured

    @pytest.mark.asyncio
    async def test_direct_anthropic_without_max_tokens_gets_model_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = litellm.get_model_info(_DIRECT_CLAUDE_MODEL)["max_output_tokens"]
        assert isinstance(expected, int) and expected > 4096  # fixture sanity

        captured = await _capture_kwargs(monkeypatch, model=_DIRECT_CLAUDE_MODEL, params=None)

        assert captured["max_tokens"] == expected

    @pytest.mark.asyncio
    async def test_openrouter_via_custom_llm_provider_override_is_also_excluded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exclusion must follow LiteLLM's dispatch, not just the prefix.

        ``custom_llm_provider`` outranks the model string when LiteLLM picks a
        provider, so this call reaches OpenRouter — and would 402 — while
        ``model_descriptor`` reads the bare string as direct Anthropic and the
        prefix test alone does not fire.
        """
        assert litellm.get_llm_provider(_DIRECT_CLAUDE_MODEL)[1] == "anthropic"
        assert (
            litellm.get_llm_provider(_DIRECT_CLAUDE_MODEL, custom_llm_provider="openrouter")[1]
            == "openrouter"
        )

        captured = await _capture_kwargs(
            monkeypatch,
            model=_DIRECT_CLAUDE_MODEL,
            params={"custom_llm_provider": "openrouter"},
        )

        assert "max_tokens" not in captured

    @pytest.mark.asyncio
    async def test_explicit_caller_max_tokens_wins_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQUIRED TEST 2 — an agent-supplied ``max_tokens`` is honoured verbatim.

        The 52 agents mitigated by hand carry explicit per-model values; the
        central default must never overwrite them.
        """
        captured = await _capture_kwargs(
            monkeypatch,
            # Direct, not OpenRouter: this must fail if the precedence guard is
            # removed, and on OpenRouter the exclusion would mask that.
            model=_DIRECT_CLAUDE_MODEL,
            params={"max_tokens": 1234, "thinking": {"type": "adaptive"}},
        )

        assert captured["max_tokens"] == 1234

    @pytest.mark.asyncio
    async def test_explicit_max_completion_tokens_also_suppresses_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``max_completion_tokens`` is the same reservation under OpenAI's newer
        spelling (litellm maps it onto ``max_tokens``). Injecting our own
        ``max_tokens`` alongside it would send two competing caps.
        """
        captured = await _capture_kwargs(
            monkeypatch,
            # Direct, so the absence below is attributable to this guard rather
            # than to the OpenRouter exclusion.
            model=_DIRECT_CLAUDE_MODEL,
            params={"max_completion_tokens": 4321},
        )

        assert "max_tokens" not in captured
        assert captured["max_completion_tokens"] == 4321

    @pytest.mark.asyncio
    async def test_explicit_max_output_tokens_is_the_only_cap_and_is_the_callers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQUIRED TEST — ``max_output_tokens`` suppresses the default too.

        Third accepted spelling of a caller cap. ``output_reservation()``
        already honored it for windowing, so before this fix the harness
        reserved 1234 locally while sending the model ceiling on the wire:
        two competing limits, and the provider obeyed the larger one.

        The assertion is deliberately "exactly ONE output cap, and it is the
        caller's" rather than "``max_output_tokens`` is preserved verbatim".
        Preserving it verbatim is what the wire probe shows to be broken: see
        ``test_real_litellm_path_sends_only_the_callers_max_output_tokens``.
        """
        captured = await _capture_kwargs(
            monkeypatch,
            # Direct route: the default would otherwise fire here, so a
            # suppression failure is attributable to this guard alone.
            model=_DIRECT_CLAUDE_MODEL,
            params={"max_output_tokens": 1234, "thinking": {"type": "adaptive"}},
        )

        caps = {
            key: captured[key]
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if key in captured
        }
        assert len(caps) == 1, f"expected exactly one output cap, got {caps}"
        assert next(iter(caps.values())) == 1234

    @pytest.mark.asyncio
    async def test_unknown_model_omits_max_tokens_rather_than_sending_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQUIRED TEST 3 — ``get_model_info`` raising must fail safe.

        ``max_tokens: None`` is not the same as absent: litellm forwards an
        explicit ``None`` into the provider body on some adapters, which is a
        400 rather than a fallback to the provider default.
        """
        captured = await _capture_kwargs(
            monkeypatch,
            # Unmapped but unmistakably Claude-shaped, so the Anthropic gate
            # admits it and only the catalog lookup can fail.
            model="anthropic/claude-not-a-real-model-2451",
            params={"thinking": {"type": "adaptive"}},
        )

        assert "max_tokens" not in captured

    @pytest.mark.asyncio
    async def test_model_info_without_max_output_tokens_omits_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQUIRED TEST 3 (second arm) — a catalog entry that exists but carries
        no ``max_output_tokens`` must also degrade to *absent*, never ``None``.
        """
        monkeypatch.setattr(
            litellm,
            "get_model_info",
            lambda model: {"max_input_tokens": 200_000, "max_output_tokens": None},
        )
        completion.default_max_output_tokens.cache_clear()

        captured = await _capture_kwargs(
            monkeypatch, model=_DIRECT_CLAUDE_MODEL, params={"thinking": {"type": "adaptive"}}
        )

        assert "max_tokens" not in captured
        completion.default_max_output_tokens.cache_clear()

    @pytest.mark.asyncio
    async def test_openai_route_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default is deliberately scoped to Anthropic-shaped routes.

        OpenAI's no-``max_tokens`` behaviour is already "as much as fits", so
        there is no defect to fix — and reserving the full output ceiling there
        is an active regression: OpenAI rejects a request whose prompt plus
        ``max_tokens`` exceeds the context window, and OpenRouter 402s when
        ``max_tokens`` exceeds a key's remaining credit affordance (the failure
        ``evals/wam_fusion/recipes.py`` already caps around).
        """
        captured = await _capture_kwargs(
            monkeypatch, model="openai/gpt-4o", params={"temperature": 0.5}
        )

        assert "max_tokens" not in captured

    @pytest.mark.asyncio
    async def test_streaming_path_gets_the_same_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both wrappers build kwargs through ``_build_litellm_kwargs``; pin that
        the streaming path is covered so the defect cannot survive on one arm.
        """
        kwargs = completion._build_litellm_kwargs(
            model=_DIRECT_CLAUDE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            auth=None,
            extra=None,
            session_id="sess_1",
            stream=True,
        )

        assert (
            kwargs["max_tokens"]
            == litellm.get_model_info(_DIRECT_CLAUDE_MODEL)["max_output_tokens"]
        )


class TestDefaultMaxOutputTokensHelper:
    """The resolver itself: pure function of the model string, fail-safe."""

    def test_resolves_known_claude_ceiling(self) -> None:
        assert completion.default_max_output_tokens("anthropic/claude-sonnet-4-5") == 64_000

    def test_unknown_model_is_none(self) -> None:
        assert completion.default_max_output_tokens("nope/not-a-model-2451") is None

    def test_raising_lookup_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(model: str) -> dict[str, Any]:
            raise RuntimeError("catalog exploded")

        monkeypatch.setattr(litellm, "get_model_info", boom)
        completion.default_max_output_tokens.cache_clear()
        try:
            assert completion.default_max_output_tokens("anthropic/claude-sonnet-4-5") is None
        finally:
            completion.default_max_output_tokens.cache_clear()

    def test_non_positive_ceiling_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A zero/negative catalog value is data corruption, not a budget."""
        monkeypatch.setattr(litellm, "get_model_info", lambda model: {"max_output_tokens": 0})
        completion.default_max_output_tokens.cache_clear()
        try:
            assert completion.default_max_output_tokens("anthropic/claude-sonnet-4-5") is None
        finally:
            completion.default_max_output_tokens.cache_clear()


class TestRealLiteLLMWireBody:
    """Evidence through the REAL litellm path, not a stub that bypasses it.

    ``_capture_kwargs`` monkeypatches ``litellm.acompletion``, so it observes
    what aios *hands to* litellm — it cannot see the defaulting litellm then
    applies on top. That blind spot is exactly where this defect lived:
    ``AnthropicConfig.get_config()`` injects ``max_tokens`` when
    ``get_optional_params`` left it unset, so a request aios believed carried
    only the caller's cap arrived at Anthropic carrying the model ceiling too.

    These tests intercept at the httpx transport instead, so the assertion is
    about the bytes on the wire.
    """

    @staticmethod
    def _wire_body(monkeypatch: pytest.MonkeyPatch, *, model: str, params: dict[str, Any]) -> Any:
        import asyncio
        import json

        import httpx
        import litellm.llms.custom_httpx.http_handler as hh

        captured: dict[str, Any] = {}

        class _Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-5-5",
                        "content": [{"type": "text", "text": "hi"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    },
                )

        client = httpx.AsyncClient(transport=_Transport())
        original_init = hh.AsyncHTTPHandler.__init__

        def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.client = client

        monkeypatch.setattr(hh.AsyncHTTPHandler, "__init__", _patched_init)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
        # litellm memoizes provider HTTP clients, so a handler built by an
        # EARLIER call (patched or not) is reused and this transport never
        # sees the request — the body then comes back empty. Harmless while
        # each test called this helper once; flushing makes repeated calls
        # within one test, and ordering between tests, both reliable.
        litellm.in_memory_llm_clients_cache.flush_cache()  # type: ignore[no-untyped-call]

        asyncio.get_event_loop_policy()
        asyncio.run(
            completion.call_litellm(
                completion.LlmRequest(
                    messages=[{"role": "user", "content": "hi"}],
                    params=dict(params),
                    session_id="sess_wire",
                ),
                model=model,
            )
        )
        return captured["body"]

    def test_real_litellm_path_sends_only_the_callers_max_output_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller's cap must be the ONE cap Anthropic actually receives.

        Before the fix this body was
        ``{"max_output_tokens": 1234, "max_tokens": 128000}`` — litellm passes
        ``max_output_tokens`` through unrecognized and fills ``max_tokens``
        from the catalog, so the provider honored 128000 and the caller's 1234
        did nothing. Suppressing only the *harness* default does not fix that;
        the spelling has to be folded onto ``max_tokens``.
        """
        body = self._wire_body(
            monkeypatch,
            model="anthropic/claude-opus-5-5",
            params={"max_output_tokens": 1234},
        )

        assert body["max_tokens"] == 1234
        assert "max_output_tokens" not in body

    def test_real_litellm_path_reserves_the_ceiling_when_no_cap_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative control: with no caller cap the full ceiling is still sent.

        Without this arm the test above would also pass if the fix wrongly
        dropped every output cap.

        The expected value is read from the catalog rather than hardcoded on
        purpose. Under pytest, egress is blocked, so litellm cannot fetch its
        remote cost map and falls back to the bundled backup — which carries a
        *different* ceiling for this model than the live map (64000 vs
        128000). Pinning a literal here would assert the sandbox's catalog
        snapshot, not the behavior under test, and would break on any litellm
        bump.
        """
        model = "anthropic/claude-opus-5-5"
        expected = litellm.get_model_info(model)["max_output_tokens"]
        assert isinstance(expected, int) and expected > 0

        body = self._wire_body(
            monkeypatch,
            model=model,
            params={"thinking": {"type": "adaptive"}},
        )

        assert body["max_tokens"] == expected
        # Not the 4096 provider default, which is the defect this PR removes.
        assert body["max_tokens"] != 4096

    def test_real_litellm_path_never_puts_an_invalid_cap_on_the_wire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The #2453 P1 property asserted against the actual request BYTES.

        ``_capture_kwargs`` sees what aios hands litellm, not what litellm then
        sends — and that blind spot is where this defect class lives. Two of
        these values are only dangerous *after* litellm's mapping:
        ``max_completion_tokens`` is mapped onto ``max_tokens``, so a surviving
        ``-5`` becomes the cap the provider reads, and an unset ``max_tokens``
        is refilled by ``AnthropicConfig.get_config()`` with its 4096 default.

        Asserted relative to the catalog (see the sibling test) because the
        pytest sandbox blocks egress and litellm falls back to its bundled map.
        """
        model = "anthropic/claude-opus-5-5"
        expected = litellm.get_model_info(model)["max_output_tokens"]
        assert isinstance(expected, int) and expected > 0

        for params in (
            {"max_tokens": None},
            {"max_tokens": 0},
            {"max_tokens": False},
            {"max_completion_tokens": -5},
            {"max_output_tokens": "x"},
        ):
            body = self._wire_body(monkeypatch, model=model, params=dict(params))

            assert body["max_tokens"] == expected, params
            # Not the provider's silent 4096 fallback — the #2451 defect.
            assert body["max_tokens"] != 4096, params
            assert "max_output_tokens" not in body, params
            assert "max_completion_tokens" not in body, params


class TestInvalidExplicitCapFallsBackToTheDefault:
    """A cap is a POSITIVE INT — key presence alone is not a cap (#2453 P1).

    The gate that decides whether to inject the harness ceiling used to test
    ``key in params``, while ``context_budget.output_reservation`` and
    ``context_admission._output_reserve`` both required a positive int. Sharing
    the spelling *list* across the three surfaces had not unified the *validity*
    rule, so ``{"max_tokens": None}`` (and ``0`` / ``False`` / ``-5`` / ``"x"``)
    read as "the caller capped it" in exactly one of the three.

    The consequences were concrete, not theoretical: the model-ceiling default
    was suppressed, windowing reserved zero tokens, and the request reached the
    provider either carrying a value it rejects (400) or carrying nothing — in
    which case LiteLLM's ``DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS = 4096`` fallback
    stands, which is the silent-truncation defect #2451 exists to close.

    **Documented semantics, asserted here: an unusable explicit value is DROPPED
    and the model-ceiling default applies**, i.e. the request behaves exactly as
    if the caller had omitted the key. See ``_has_explicit_output_cap`` for why
    that beats forwarding it for the provider to reject.

    Every case runs on ``anthropic/*`` because that is the only route family the
    default fires on; on OpenRouter the exclusion alone would satisfy these
    assertions vacuously.
    """

    _INVALID_CAPS: ClassVar[list[dict[str, Any]]] = [
        {"max_tokens": None},
        {"max_tokens": 0},
        {"max_tokens": False},
        {"max_completion_tokens": -5},
        {"max_output_tokens": "x"},
    ]

    @staticmethod
    def _ids(params: dict[str, Any]) -> str:
        key, value = next(iter(params.items()))
        return f"{key}={value!r}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params", _INVALID_CAPS, ids=_ids)
    async def test_invalid_cap_still_gets_the_model_ceiling_default(
        self, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
    ) -> None:
        """REQUIRED TEST — red on df912388 for all five spellings/values."""
        expected = litellm.get_model_info(_DIRECT_CLAUDE_MODEL)["max_output_tokens"]
        assert isinstance(expected, int) and expected > 4096  # fixture sanity

        captured = await _capture_kwargs(
            monkeypatch, model=_DIRECT_CLAUDE_MODEL, params=dict(params)
        )

        assert captured["max_tokens"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params", _INVALID_CAPS, ids=_ids)
    async def test_no_invalid_cap_reaches_the_wire(
        self, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
    ) -> None:
        """The second half of the property, and the one a "just fix the gate"
        change would miss: applying the default is not enough if the unusable
        value travels alongside it.

        ``max_completion_tokens`` is mapped onto ``max_tokens`` by LiteLLM, so a
        surviving ``{"max_completion_tokens": -5}`` would put TWO caps on the
        wire with the invalid one winning. Assert exactly one output cap, and
        that it is the ceiling.
        """
        expected = litellm.get_model_info(_DIRECT_CLAUDE_MODEL)["max_output_tokens"]

        captured = await _capture_kwargs(
            monkeypatch, model=_DIRECT_CLAUDE_MODEL, params=dict(params)
        )

        caps = {
            key: captured[key]
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if key in captured
        }
        assert len(caps) == 1, f"expected exactly one output cap, got {caps}"
        assert next(iter(caps.values())) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "params",
        [
            {"max_tokens": 1234},
            {"max_completion_tokens": 1234},
            {"max_output_tokens": 1234},
        ],
        ids=lambda p: next(iter(p)),
    )
    async def test_a_valid_cap_still_wins_under_every_spelling(
        self, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
    ) -> None:
        """REQUIRED regression guard. Tightening the predicate from "the key is
        present" to "the value is a positive int" must not cost a VALID caller
        cap its precedence under any spelling — that would re-break #2451's
        headline property (the 52 hand-mitigated agents carry explicit values).
        """
        captured = await _capture_kwargs(
            monkeypatch, model=_DIRECT_CLAUDE_MODEL, params=dict(params)
        )

        caps = {
            key: captured[key]
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if key in captured
        }
        assert len(caps) == 1, f"expected exactly one output cap, got {caps}"
        assert next(iter(caps.values())) == 1234

    def test_the_three_surfaces_share_ONE_cap_predicate(self) -> None:
        """The drift guard, as an identity assert rather than an equality one.

        The spelling list was already shared when this defect was found; the
        VALIDITY rule was not, and that was enough to keep the three surfaces
        disagreeing. Pin that all three route through the same function object,
        so a re-introduced local predicate — even one that happens to be correct
        the day it is written, which is how this bug was born — fails here.
        """
        from aios.harness import context_admission, context_budget

        assert completion.explicit_output_cap is context_budget.explicit_output_cap
        assert context_admission.explicit_output_cap is context_budget.explicit_output_cap
        assert completion.is_output_cap_value is context_budget.is_output_cap_value

        # And the predicate itself answers the validity question one way.
        bad_values: tuple[Any, ...] = (None, 0, False, True, -5, "x", 1.5, [], {})
        for bad in bad_values:
            assert context_budget.is_output_cap_value(bad) is False, bad
        for good in (1, 1234, 128_000):
            assert context_budget.is_output_cap_value(good) is True, good

    @pytest.mark.parametrize(
        "params",
        [
            {"max_tokens": None},
            {"max_tokens": 0},
            {"max_tokens": False},
            {"max_completion_tokens": -5},
            {"max_output_tokens": "x"},
        ],
        ids=_ids,
    )
    def test_all_three_readers_agree_an_invalid_value_is_no_cap(
        self, params: dict[str, Any]
    ) -> None:
        """The property stated directly over the three readers, independent of
        any route: they must return the SAME verdict for the same input.

        This is the assertion that would have caught the original drift without
        anyone having to think of the Anthropic route, and it is why the fix is
        a shared parser rather than a second copy of the validity rule.
        """
        from aios.harness import context_admission, context_budget

        assert completion._has_explicit_output_cap(dict(params)) is False
        assert context_budget.output_reservation(dict(params)) == 0
        assert context_admission._output_reserve(dict(params)) is None

        valid = dict(params)
        valid[next(iter(params))] = 4321
        assert completion._has_explicit_output_cap(valid) is True
        assert context_budget.output_reservation(valid) == 4321
        assert context_admission._output_reserve(valid) == 4321
