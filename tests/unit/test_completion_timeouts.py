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

import pytest

from aios.harness import completion


class _StallingResponse:
    """Async iterator that yields one chunk then awaits forever.

    Mimics a streaming model response that produced an initial token
    burst and then stalled — the failure mode our timeout fixes.
    """

    def __init__(self) -> None:
        self._first_yielded = False

    def __aiter__(self) -> _StallingResponse:
        return self

    async def __anext__(self) -> object:
        if not self._first_yielded:
            self._first_yielded = True
            return _make_chunk("hello")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


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
    ``TimeoutError`` once the per-chunk inactivity bound elapses."""
    monkeypatch.setattr(completion, "_STREAM_INACTIVITY_TIMEOUT_S", 0.1)

    async def fake_acompletion(**kwargs: object) -> _StallingResponse:
        return _StallingResponse()

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)

    with pytest.raises(TimeoutError):
        await completion.stream_litellm(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
            pool=_StubPool(),  # type: ignore[arg-type]
            session_id="sess_test",
        )


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

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        completion.litellm,
        "stream_chunk_builder",
        lambda chunks: {
            "usage": {},
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        },
    )

    await completion.stream_litellm(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        pool=_StubPool(),  # type: ignore[arg-type]
        session_id="sess_test",
    )

    assert captured["timeout"] == completion._REQUEST_TIMEOUT_S
    assert captured["stream_timeout"] == completion._STREAM_INACTIVITY_TIMEOUT_S


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

    monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)

    await completion.call_litellm(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ping"}],
        extra={"timeout": 1234.0},
    )

    assert captured["timeout"] == 1234.0
