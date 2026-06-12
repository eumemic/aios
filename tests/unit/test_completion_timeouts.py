"""Per-call timeout behavior for the LiteLLM wrappers.

The harness's zero-hang guarantee depends on every external await having a
finite ceiling.  ``stream_litellm`` is the most exposed call site: a server
that silently stops sending chunks (laptop sleep, network blip, upstream
provider stall) would, without per-chunk inactivity bounds, hang the
worker for as long as the underlying HTTP layer holds the connection open.
These tests confirm the wrapper raises ``TimeoutError`` instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import litellm
import pytest

from aios.config import Settings
from aios.harness import completion


class _RecordingResponse:
    """Async iterator that records whether ``aclose()`` was called.

    Drives the same manual ``__anext__`` path ``stream_litellm`` uses, so
    the production ``finally`` can be asserted against. ``stall`` makes the
    iterator hang forever after the scripted chunks (to exercise the
    inter-chunk timeout); ``raise_after`` makes ``__anext__`` raise the
    given exception after the scripted chunks (to exercise a mid-stream
    error). With neither, the iterator drains normally via
    ``StopAsyncIteration``.
    """

    def __init__(
        self,
        chunks: list[object],
        *,
        stall: bool = False,
        raise_after: BaseException | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._stall = stall
        self._raise_after = raise_after
        self._i = 0
        self.aclose_count = 0

    def __aiter__(self) -> _RecordingResponse:
        return self

    async def __anext__(self) -> object:
        if self._i < len(self._chunks):
            chunk = self._chunks[self._i]
            self._i += 1
            return chunk
        if self._stall:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        if self._raise_after is not None:
            raise self._raise_after
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_count += 1


async def _slow_first_chunk_response(ttft_delay_s: float) -> AsyncIterator[object]:
    """Delay the first chunk by ``ttft_delay_s``, yield it, then exit.

    Mimics cold-cache long-prompt streaming: TTFT exceeds the inter-chunk
    bound, but once streaming starts there's no abnormal gap — the failure
    shape issue #239 fixes.
    """
    await asyncio.sleep(ttft_delay_s)
    yield _make_chunk("hello")


def _make_chunk(text: str | None) -> object:
    """Build the minimum shape ``stream_litellm`` reads off each chunk."""
    from types import SimpleNamespace

    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


class _StubPool:
    """``stream_litellm`` only uses the pool to ``pg_notify`` deltas.

    A no-op pool keeps the chunk loop honest — every yielded chunk runs
    through ``_notify_delta`` which pulls a connection.
    """

    def acquire(self) -> _StubPoolAcquire:
        return _StubPoolAcquire()


class _StubPoolAcquire:
    async def __aenter__(self) -> _StubConnection:
        return _StubConnection()

    async def __aexit__(self, *args: object) -> None:
        pass


class _StubConnection:
    async def execute(self, *args: object) -> None:
        pass


@pytest.mark.asyncio
async def test_stream_litellm_raises_timeout_on_stalled_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming response that hangs after the first chunk must raise
    ``TimeoutError`` once the per-chunk inactivity bound elapses.

    Also verifies that when the inter-chunk guard fires, the litellm stream
    wrapper is closed (``aclose`` called) before ``TimeoutError`` propagates,
    so the underlying httpx socket is released rather than leaked until GC
    (issue #855)."""
    monkeypatch.setattr(completion, "_STREAM_INTER_CHUNK_TIMEOUT_S", 0.1)

    resp = _RecordingResponse([_make_chunk("hello")], stall=True)

    async def fake_acompletion(**kwargs: object) -> _RecordingResponse:
        return resp

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(TimeoutError):
        await completion.stream_litellm(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
            pool=_StubPool(),
            session_id="sess_test",
        )

    assert resp.aclose_count == 1


@pytest.mark.asyncio
async def test_stream_litellm_raises_deadline_with_salvaged_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(completion, "get_settings", lambda: Settings(model_call_deadline_s=1.0))
    monkeypatch.setattr(completion, "_STREAM_TTFT_TIMEOUT_S", 1.0)
    monkeypatch.setattr(completion, "_STREAM_INTER_CHUNK_TIMEOUT_S", 1.0)

    resp = _RecordingResponse([_make_chunk("hello")], stall=True)

    async def fake_acompletion(**kwargs: object) -> _RecordingResponse:
        return resp

    def fake_builder(chunks: list[object], **kwargs: object) -> dict[str, object]:
        assert len(chunks) == 1
        return {
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", fake_builder)

    with pytest.raises(completion.ModelCallDeadlineError) as excinfo:
        await completion.stream_litellm(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
            pool=_StubPool(),
            session_id="sess_test",
        )

    assert excinfo.value.chunks_seen == 1
    assert excinfo.value.usage == {
        "input_tokens": 11,
        "output_tokens": 22,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4,
    }
    assert excinfo.value.cost_usd is None
    assert resp.aclose_count == 1


@pytest.mark.asyncio
async def test_call_litellm_deadline_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(completion, "get_settings", lambda: Settings(model_call_deadline_s=1.0))

    async def fake_acompletion(**kwargs: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(completion.ModelCallDeadlineError) as excinfo:
        await completion.call_litellm(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert excinfo.value.chunks_seen == 0
    assert excinfo.value.usage == {}
    assert excinfo.value.cost_usd is None


@pytest.mark.asyncio
async def test_stream_litellm_long_ttft_succeeds_when_inter_chunk_is_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-chunk wait may exceed the inter-chunk bound; the TTFT ceiling
    applies only to the first chunk."""
    monkeypatch.setattr(completion, "_STREAM_INTER_CHUNK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(completion, "_STREAM_TTFT_TIMEOUT_S", 2.0)

    async def fake_acompletion(**kwargs: object) -> AsyncIterator[object]:
        return _slow_first_chunk_response(ttft_delay_s=0.1)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        litellm,
        "stream_chunk_builder",
        lambda chunks: {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        },
    )

    message, _, _, _ = await completion.stream_litellm(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        pool=_StubPool(),
        session_id="sess_test",
    )

    assert message["content"] == "hello"


@pytest.mark.asyncio
async def test_stream_litellm_passes_timeout_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults must reach ``litellm.acompletion``.  This is what bounds
    the total request and the LiteLLM-side per-chunk inactivity."""
    captured: dict[str, object] = {}

    class _EmptyResponse:
        def __aiter__(self) -> _EmptyResponse:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    async def fake_acompletion(**kwargs: object) -> _EmptyResponse:
        captured.update(kwargs)
        return _EmptyResponse()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        litellm,
        "stream_chunk_builder",
        lambda chunks: {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        },
    )

    await completion.stream_litellm(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        pool=_StubPool(),
        session_id="sess_test",
    )

    assert captured["timeout"] == completion._REQUEST_TIMEOUT_S
    assert captured["stream_timeout"] == completion._STREAM_INTER_CHUNK_TIMEOUT_S


@pytest.mark.asyncio
async def test_extra_overrides_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agents can override the default request timeout via ``litellm_extra``.

    This is the documented path for legitimately long-running calls (e.g.
    Opus 4.7 against 1M-context prompts) — the default protects all
    sessions but motivated operators can raise the bound per-agent.
    """
    captured: dict[str, object] = {}

    class _DictResponse(dict[str, object]):
        """Subscriptable + ``.get`` like the real LiteLLM response dict.

        ``_hidden_params`` is set in ``__init__`` so ``_extract_cost``
        finds the empty mapping it expects — a bare dict raises
        ``AttributeError`` on the lookup.
        """

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._hidden_params: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _DictResponse:
        captured.update(kwargs)
        return _DictResponse(
            choices=[{"message": {"role": "assistant", "content": ""}}],
            usage={},
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        extra={"timeout": 1234.0},
    )

    assert captured["timeout"] == 1234.0


def _make_empty_choices_chunk() -> object:
    """Build a usage-summary chunk: ``choices=[]`` with usage metadata.

    OpenAI with ``stream_options.include_usage=True``, OpenRouter, Grok,
    and OpenAI-compatible vLLM all emit a final chunk of this shape that
    carries usage but no delta.
    """
    from types import SimpleNamespace

    return SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=42))


@pytest.mark.asyncio
async def test_stream_litellm_tolerates_empty_choices_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming response that includes a chunk with ``choices=[]`` (a
    usage-summary chunk emitted by OpenRouter/Grok/vLLM/OpenAI-with-usage)
    must not crash the stream loop."""

    async def fake_stream(**kwargs: Any) -> AsyncIterator[object]:
        yield _make_chunk("hello")
        yield _make_empty_choices_chunk()

    async def fake_acompletion(**kwargs: object) -> AsyncIterator[object]:
        return fake_stream()

    captured: dict[str, list[object]] = {}

    def fake_builder(chunks: list[object], **kwargs: Any) -> dict[str, object]:
        captured["chunks"] = list(chunks)
        return {
            "usage": {"total_tokens": 42},
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", fake_builder)

    message, _, _, _ = await completion.stream_litellm(
        model="openrouter/anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        pool=_StubPool(),
        session_id="sess_test",
    )

    assert message["content"] == "hello"
    # The usage-summary chunk must reach stream_chunk_builder so usage data
    # isn't dropped — the guard skips notify, not collection.
    assert len(captured["chunks"]) == 2
    assert captured["chunks"][-1].choices == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_litellm_raises_typed_error_on_zero_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the provider closes the connection without emitting any chunks
    (Bedrock cold start, OpenRouter mid-handshake disconnect, vLLM under
    load), ``chunks=[]`` reaches ``litellm.stream_chunk_builder`` — which
    returns ``None`` rather than raising. The current code then calls
    ``assembled.get("usage")`` which crashes with a generic
    ``AttributeError: 'NoneType' object has no attribute 'get'`` — an
    opaque Python implementation detail in operator logs that doesn't
    name the actual failure mode.

    Per CLAUDE.md ``fail hard, no fallbacks`` plus ``model sees raw
    errors``: surface a typed, descriptive error so operators see
    "provider returned empty stream" rather than a NoneType crash and
    the harness retry path logs a meaningful ``step.litellm_failed``
    reason.
    """

    class _EmptyResponse:
        def __aiter__(self) -> _EmptyResponse:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    async def fake_acompletion(**_kwargs: object) -> _EmptyResponse:
        return _EmptyResponse()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    # Intentionally do NOT patch ``stream_chunk_builder`` — we want the
    # real-world behavior where it returns ``None`` for an empty chunk list.
    # (Verified: ``litellm.stream_chunk_builder(chunks=[])`` returns None.)

    with pytest.raises(Exception) as excinfo:
        await completion.stream_litellm(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
            pool=_StubPool(),
            session_id="sess_test",
        )

    # Pre-fix: AttributeError("'NoneType' object has no attribute 'get'").
    # Post-fix: a typed error whose message names "empty" so it's grep-able
    # in operator logs.
    msg = str(excinfo.value)
    assert "NoneType" not in msg, (
        f"completion must surface a typed error for empty streams, not a "
        f"NoneType implementation detail; got {type(excinfo.value).__name__}: {msg!r}"
    )
    assert "empty" in msg.lower() or "no chunks" in msg.lower() or "zero" in msg.lower(), (
        f"error message must name the failure mode (empty/no-chunks/zero); "
        f"got {type(excinfo.value).__name__}: {msg!r}"
    )


@pytest.mark.asyncio
async def test_stream_litellm_closes_stream_on_mid_stream_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any non-StopAsyncIteration exit from the chunk loop must still close
    the stream. A provider/adapter error mid-stream must propagate, but only
    after ``aclose`` releases the connection (issue #855)."""
    boom = RuntimeError("adapter exploded mid-stream")
    resp = _RecordingResponse([_make_chunk("hello")], raise_after=boom)

    async def fake_acompletion(**kwargs: object) -> _RecordingResponse:
        return resp

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(RuntimeError, match="adapter exploded mid-stream"):
        await completion.stream_litellm(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
            pool=_StubPool(),
            session_id="sess_test",
        )

    assert resp.aclose_count == 1


@pytest.mark.asyncio
async def test_stream_litellm_closes_stream_on_normal_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normally-drained stream is also closed — the ``finally`` is
    unconditional and closing after a full drain is a harmless no-op
    (issue #855)."""
    resp = _RecordingResponse([_make_chunk("hel"), _make_chunk("lo")])

    async def fake_acompletion(**kwargs: object) -> _RecordingResponse:
        return resp

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        litellm,
        "stream_chunk_builder",
        lambda chunks: {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        },
    )

    message, _, _, _ = await completion.stream_litellm(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        pool=_StubPool(),
        session_id="sess_test",
    )

    assert message["content"] == "hello"
    assert resp.aclose_count == 1
