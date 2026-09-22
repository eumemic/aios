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
"""

from __future__ import annotations

from typing import Any

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
    async def test_explicit_caller_max_tokens_wins_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQUIRED TEST 2 — an agent-supplied ``max_tokens`` is honoured verbatim.

        The 52 agents mitigated by hand carry explicit per-model values; the
        central default must never overwrite them.
        """
        captured = await _capture_kwargs(
            monkeypatch,
            model=_PROXY_CLAUDE_MODEL,
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
            model=_PROXY_CLAUDE_MODEL,
            params={"max_completion_tokens": 4321},
        )

        assert "max_tokens" not in captured
        assert captured["max_completion_tokens"] == 4321

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
