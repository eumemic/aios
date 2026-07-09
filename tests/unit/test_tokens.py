"""Unit tests for the approx_tokens estimator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from aios.harness.context import render_user_event
from aios.harness.tokens import approx_tokens

_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class TestApproxTokens:
    """approx_tokens delegates to litellm's local tokenizer with no
    model specified.  Exact counts depend on tokenizer version, so
    these tests pin invariants rather than specific numbers.
    """

    def test_non_empty_message_costs_something(self) -> None:
        assert approx_tokens([{"role": "user", "content": "hello world"}]) >= 1

    def test_longer_content_costs_more(self) -> None:
        short = [{"role": "user", "content": "hi"}]
        long = [{"role": "user", "content": "hello, this is a longer message"}]
        assert approx_tokens(long) > approx_tokens(short)

    def test_empty_list_is_cheap(self) -> None:
        """litellm charges a small constant for chat framing even on an
        empty list; we just need it not to raise and not to explode.
        """
        assert 0 <= approx_tokens([]) < 10

    def test_tool_calls_increase_cost(self) -> None:
        plain = [{"role": "assistant", "content": ""}]
        with_call = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            }
        ]
        assert approx_tokens(with_call) > approx_tokens(plain)

    def test_multi_message_list_sums(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        assert approx_tokens(msgs) > approx_tokens(msgs[:1])
        assert approx_tokens(msgs) > approx_tokens(msgs[:2])


class TestApproxTokensUnderFocalRendering:
    """The cumulative_tokens column is computed against the as-rendered
    form of user events.  A non-focal inbound that renders as a short
    notification marker must cost far fewer tokens than the same inbound
    rendered with full content + metadata header.  This invariant keeps
    the chunked-window slicer honest even in busy many-to-one sessions.
    """

    _CHAN = "signal/bot/alice"

    def test_notification_smaller_than_full(self) -> None:
        long_text = "this is a moderately long message body " * 10
        metadata = {
            "channel": self._CHAN,
            "sender_uuid": "u1",
            "sender_name": "Alice",
            "timestamp_ms": 1776401210703,
        }
        data = {"role": "user", "content": long_text, "metadata": metadata}

        focal_render = render_user_event(data, self._CHAN, self._CHAN, _CREATED_AT)
        notif_render = render_user_event(data, self._CHAN, "signal/bot/other", _CREATED_AT)

        assert approx_tokens([notif_render]) < approx_tokens([focal_render])

    def test_legacy_null_render_adds_received_envelope(self) -> None:
        """orig=None now carries the uniform ``received=`` envelope, so the
        rendered form costs a few tokens more than the raw data (was: equal
        — the channel=None path used to pass content through untouched)."""
        data = {"role": "user", "content": "hello"}
        rendered = render_user_event(data, None, None, _CREATED_AT)
        assert rendered["content"].startswith("[received=")
        assert approx_tokens([rendered]) > approx_tokens([data])


class TestApproxTokensWithTools:
    """The ``tools=`` kwarg exists so the ``model_request_end`` span-stamp
    call site can cost the exact payload the provider sees (messages +
    tools).

    The two existing call sites in ``db/queries.py`` (per-event
    cumulative_tokens) and ``tools/switch_channel.py`` (recap budget) hand
    messages only — these tests pin that the default behavior is
    byte-identical whether the kwarg is omitted, None, or [].
    """

    _MSGS: ClassVar[list[dict[str, Any]]] = [{"role": "user", "content": "what files are here"}]
    _TOOL: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }

    def test_tools_increase_cost(self) -> None:
        assert approx_tokens(self._MSGS, tools=[self._TOOL]) > approx_tokens(self._MSGS)

    def test_tools_none_identical_to_omitted(self) -> None:
        assert approx_tokens(self._MSGS, tools=None) == approx_tokens(self._MSGS)

    def test_tools_empty_list_identical_to_omitted(self) -> None:
        assert approx_tokens(self._MSGS, tools=[]) == approx_tokens(self._MSGS)


class TestApproxTokensEquivalenceMatrix:
    """Issue #1744: the memoized primitives must not change what
    ``approx_tokens`` returns.  Every slate here is checked against a
    direct, unmemoized ``litellm.token_counter(messages=..., tools=...)``
    call — the ground truth the old implementation delegated to directly.
    """

    _TOOL: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }

    _SYSTEM: ClassVar[dict[str, Any]] = {"role": "system", "content": "you are a helpful agent"}
    _USER: ClassVar[dict[str, Any]] = {"role": "user", "content": "what files are here"}
    _TOOL_CALL_MSG: ClassVar[dict[str, Any]] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
    }
    _REASONING_MSG: ClassVar[dict[str, Any]] = {
        "role": "assistant",
        "content": "the answer",
        "reasoning_content": "a long internal chain of deliberation",
    }
    _NAME_KEY_MSG: ClassVar[dict[str, Any]] = {
        "role": "user",
        "name": "alice",
        "content": "hi there",
    }
    _LIST_CONTENT_MSG: ClassVar[dict[str, Any]] = {
        "role": "user",
        "content": [{"type": "text", "text": "hello from a list"}],
    }
    _TOOL_RESULT_MSG: ClassVar[dict[str, Any]] = {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": "result payload text",
    }

    def _slates(self) -> list[list[dict[str, Any]]]:
        return [
            [],
            [self._USER],
            [self._SYSTEM, self._USER],
            [self._SYSTEM, self._USER, self._TOOL_CALL_MSG],
            [self._REASONING_MSG],
            [self._NAME_KEY_MSG],
            [self._LIST_CONTENT_MSG],
            [self._TOOL_RESULT_MSG],
            [
                self._SYSTEM,
                self._USER,
                self._REASONING_MSG,
                self._TOOL_CALL_MSG,
                self._TOOL_RESULT_MSG,
                self._NAME_KEY_MSG,
                self._LIST_CONTENT_MSG,
            ],
        ]

    def test_matches_direct_token_counter(self) -> None:
        from litellm import token_counter

        for tools in (None, [self._TOOL]):
            for msgs in self._slates():
                got = approx_tokens(msgs, tools=tools)
                want = int(
                    token_counter(
                        messages=list(msgs),
                        tools=list(tools) if tools else None,
                    )
                )
                assert got == want, (msgs, tools, got, want)


class TestApproxTokensCache:
    """Issue #1744: memoization must be transparent — same results, but
    the underlying ``token_counter`` isn't re-invoked on a repeat call
    with fresh-but-equal dicts.
    """

    def test_repeat_call_makes_zero_token_counter_calls(self, monkeypatch: Any) -> None:
        msgs = [
            {"role": "user", "content": "warm the cache please, this is unique-9182"},
            {"role": "assistant", "content": "sure thing"},
        ]
        # Warm the cache first (fresh dicts).
        first = approx_tokens([dict(m) for m in msgs])

        calls = {"n": 0}
        real_token_counter = None
        import litellm

        real_token_counter = litellm.token_counter

        def _counting(*args: Any, **kwargs: Any) -> int:
            calls["n"] += 1
            return int(real_token_counter(*args, **kwargs))

        monkeypatch.setattr(litellm, "token_counter", _counting)

        # Fresh-but-equal dicts (new objects, same content) must hit cache.
        second = approx_tokens([dict(m) for m in msgs])

        assert second == first
        assert calls["n"] == 0

    def test_changed_message_is_recounted(self) -> None:
        a = [{"role": "user", "content": "version A of the message"}]
        b = [{"role": "user", "content": "version B of the message, different"}]
        assert approx_tokens(a) != approx_tokens(b)

    def test_unserializable_payload_still_counted(self) -> None:
        class Weird:
            def __repr__(self) -> str:
                return "Weird()"

        # A message value that json.dumps can't natively serialize (falls
        # back to `default=repr`, or fails outright) must still be counted,
        # not raise.
        msg = {"role": "user", "content": "hello", "_marker": Weird()}
        result = approx_tokens([msg])
        assert result >= 1

    def test_cache_bound_enforced(self) -> None:
        from aios.harness.tokens import _BODY_CACHE, _BODY_CACHE_MAX

        # Blow well past the cache bound with unique messages; the cache
        # must never exceed its max size.
        for i in range(200):
            approx_tokens([{"role": "user", "content": f"unique message body {i}"}])
        assert len(_BODY_CACHE) <= _BODY_CACHE_MAX


class TestApproxTokensThreaded:
    """Issue #1744: concurrent calls (as the loop.py ``to_thread`` call
    site will make) must return identical, correct results — no shared
    mutable state races.
    """

    def test_concurrent_calls_agree(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": "you are a helpful agent"},
            {"role": "user", "content": "what is the weather like today in paris"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city": "paris"}'},
                    }
                ],
            },
        ]
        expected = approx_tokens(msgs)

        def _call(_: int) -> int:
            return approx_tokens([dict(m) for m in msgs])

        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(_call, range(64)))

        assert all(r == expected for r in results)


class TestEventTokenDelta:
    """``_event_token_delta`` is the pure per-event token contribution that
    ``append_event`` now computes BEFORE the row lock (issue #862).

    Each case asserts the helper EQUALS the inline old-form computation that
    used to live inside the transaction — equality of the two forms, not an
    absolute magnitude (tokenizer counts are tokenizer-version dependent).
    The user branch reproduces ``render_user_event(...) + separator`` exactly;
    every other branch is ``approx_tokens([data])``.
    """

    @staticmethod
    def _old_form_user(data: dict[str, Any], orig_channel: str | None, focal: str | None) -> int:
        from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT

        rendered = render_user_event(data, orig_channel, focal, _CREATED_AT)
        separator = {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT}
        return approx_tokens([rendered, separator])

    def _delta(
        self,
        kind: str,
        data: dict[str, Any],
        orig_channel: str | None = None,
        focal: str | None = None,
    ) -> int:
        from aios.db.queries.events import _event_token_delta

        return _event_token_delta(kind, data, orig_channel, focal)

    def test_delta_user_channel_none(self) -> None:
        data = {"role": "user", "content": "hello there"}
        assert self._delta("message", data) == self._old_form_user(data, None, None)

    def test_delta_user_on_channel(self) -> None:
        # orig_channel == focal: full-content render
        data = {"role": "user", "content": "what files are here", "metadata": {"from": "+1"}}
        assert self._delta("message", data, "tg:42", "tg:42") == self._old_form_user(
            data, "tg:42", "tg:42"
        )

    def test_delta_user_off_channel(self) -> None:
        # orig_channel != focal: off-channel notification marker (far fewer tokens)
        data = {"role": "user", "content": "ping from another chat", "metadata": {"from": "+1"}}
        assert self._delta("message", data, "tg:7", "tg:42") == self._old_form_user(
            data, "tg:7", "tg:42"
        )

    def test_delta_assistant(self) -> None:
        data = {"role": "assistant", "content": "sure, on it"}
        assert self._delta("message", data) == approx_tokens([data])

    def test_delta_tool(self) -> None:
        data = {"role": "tool", "tool_call_id": "tc_1", "name": "bash", "content": "ok"}
        assert self._delta("message", data) == approx_tokens([data])

    def test_delta_oversized(self) -> None:
        data = {"role": "assistant", "content": "x" * 100_000}
        assert self._delta("message", data) == approx_tokens([data])

    def test_delta_non_message_is_zero(self) -> None:
        assert self._delta("span", {"event": "sweep_start"}) == 0
        assert self._delta("lifecycle", {"event": "turn_ended"}) == 0


class TestPrecomputeEventAppend:
    """``precompute_event_append`` (issue #991) packages the pre-lock compute
    into a ``_PrecomputedAppend`` so the two tool-result appenders can run it
    OUTSIDE their outer ``FOR UPDATE``.  Its ``token_delta`` must EQUAL the
    legacy inline ``_event_token_delta`` value — including the ~100 KB payload —
    and its ``resolved_tool_channel`` must echo a supplied ``tool_parent_channel``
    (the hot path) without touching the DB.
    """

    async def test_tool_delta_matches_event_token_delta_incl_oversized(self) -> None:
        from aios.db.queries.events import _event_token_delta, precompute_event_append

        data = {
            "role": "tool",
            "tool_call_id": "tc_1",
            "name": "bash",
            "content": "x" * 100_000,
        }
        # A tool event with an explicit ``tool_parent_channel`` does NO DB I/O:
        # the delta is pure ``approx_tokens([data])`` and the channel is echoed.
        # ``conn=None`` proves the path never touches the connection.
        result = await precompute_event_append(
            None,
            account_id="acc_1",
            session_id="ses_1",
            kind="message",
            data=data,
            tool_parent_channel="tg:42",
        )
        assert result.token_delta == _event_token_delta("message", data, None, None)
        assert result.resolved_tool_channel == "tg:42"

    async def test_non_message_is_zero_delta_no_io(self) -> None:
        from aios.db.queries.events import precompute_event_append

        result = await precompute_event_append(
            None,
            account_id="acc_1",
            session_id="ses_1",
            kind="span",
            data={"event": "sweep_start"},
        )
        assert result.token_delta == 0
        assert result.resolved_tool_channel is None
