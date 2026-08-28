"""Unit tests for the MCP auto-review checker (``harness/auto_review.py``).

The deterministic half of jarbot#229's eval list lives here: timeout → ask,
junk → ask, deny-shaped → ask (coerced), one retry inside the budget,
fail-closed reason copy, allow → confirmed + woken, ask → load-bearing
``tool_requested`` marker with the reason, interrupt-invalidated allow
dropped, marker idempotence. The model-judgment half (surprise recipient →
ask, etc.) is the ``evals/auto_review`` corpus, run against the real model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from aios.errors import ConflictError
from aios.harness import auto_review, runtime
from aios.harness.auto_review import (
    AUTO_REVIEW_SOURCE,
    AUTO_REVIEW_SPAN_EVENT,
    CHECKER_UNAVAILABLE_REASON,
    REVIEW_KEY_PREFIX,
    _build_messages,
    _parse_verdict,
    _recent_user_lines,
    launch_auto_review,
)
from aios.services import model_providers as model_providers_service
from aios.services import sessions as sessions_service

# ── fixtures ────────────────────────────────────────────────────────────────


def _settings(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "auto_review_model": "openai/responses/gpt-5.6-luna",
        "auto_review_timeout_s": 5.0,
        "inference_credential_policy": "legacy_env",
        "tenancy_posture": "internal",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _surface(description: str | None = "Accountant. Keeps the books.") -> Any:
    return SimpleNamespace(description=description, tools=[], mcp_servers=[])


def _call(call_id: str = "tc_1", name: str = "mcp__gmail__send_email") -> dict[str, Any]:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": '{"to": "a@b.c", "body": "hi"}'},
    }


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        usage={"input_tokens": 100, "output_tokens": 20},
        cost=0.0001,
    )


@dataclass
class _Recorder:
    """Captures every session-service write the module performs."""

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    confirms: list[dict[str, Any]] = field(default_factory=list)
    wakes: list[str] = field(default_factory=list)
    usage_increments: list[dict[str, Any]] = field(default_factory=list)
    marker_exists: bool = False
    interrupt_seqs: list[int | None] = field(default_factory=lambda: [None, None])
    confirm_error: Exception | None = None

    def patches(self) -> list[Any]:
        async def append_event(
            pool: Any, sid: str, kind: str, data: dict[str, Any], *, account_id: str
        ) -> SimpleNamespace:
            self.events.append((kind, data))
            return SimpleNamespace(id=f"ev_{len(self.events)}", seq=len(self.events))

        async def confirm_tool_allow(
            pool: Any, sid: str, tcid: str, *, account_id: str, source: str | None = None
        ) -> SimpleNamespace:
            if self.confirm_error is not None:
                raise self.confirm_error
            self.confirms.append({"tool_call_id": tcid, "source": source})
            return SimpleNamespace(id="confirm_ev")

        async def defer_wake(pool: Any, sid: str, *, cause: str, account_id: str) -> None:
            self.wakes.append(cause)

        async def increment_usage(pool: Any, sid: str, *, account_id: str, **kw: Any) -> int:
            self.usage_increments.append(dict(kw))
            return 0

        async def has_marker(pool: Any, sid: str, tcid: str, *, account_id: str) -> bool:
            return self.marker_exists

        async def find_latest_interrupt_seq(pool: Any, sid: str, *, account_id: str) -> int | None:
            return self.interrupt_seqs.pop(0) if self.interrupt_seqs else None

        async def read_events(pool: Any, sid: str, **kw: Any) -> list[Any]:
            return [SimpleNamespace(data={"role": "user", "content": "send the invoice to a@b.c"})]

        sess = sessions_service
        return [
            mock.patch.object(sess, "append_event", append_event),
            mock.patch.object(sess, "confirm_tool_allow", confirm_tool_allow),
            mock.patch.object(sess, "increment_usage", increment_usage),
            mock.patch.object(sess, "has_tool_requested_marker", has_marker),
            mock.patch.object(sess, "find_latest_interrupt_seq", find_latest_interrupt_seq),
            mock.patch.object(sess, "read_events", read_events),
            mock.patch.object(auto_review, "defer_wake", defer_wake),
        ]

    def spans(self, event: str) -> list[dict[str, Any]]:
        return [d for k, d in self.events if k == "span" and d.get("event") == event]

    def markers(self) -> list[dict[str, Any]]:
        return [
            d for k, d in self.events if k == "lifecycle" and d.get("event") == "tool_requested"
        ]


_ENV_AUTH = (SimpleNamespace(kind="env"), None)


async def _run_review(
    rec: _Recorder,
    *,
    model_results: list[Any],
    settings: SimpleNamespace | None = None,
    auth_result: Any = _ENV_AUTH,
    call: dict[str, Any] | None = None,
    surface: Any = None,
) -> None:
    """Drive ``_review_tool_call`` with everything below it faked."""
    calls_made: list[Any] = []

    async def fake_call_litellm(request: Any, *, model: str, auth: Any) -> Any:
        calls_made.append(request)
        result = model_results.pop(0)
        if isinstance(result, Exception):
            raise result
        if result == "hang":
            await asyncio.sleep(3600)
        return result

    async def fake_auth(*a: Any, **kw: Any) -> Any:
        if isinstance(auth_result, Exception):
            raise auth_result
        return auth_result

    patches = [
        *rec.patches(),
        mock.patch.object(auto_review, "call_litellm", fake_call_litellm),
        mock.patch.object(auto_review, "get_settings", lambda: settings or _settings()),
        mock.patch.object(
            model_providers_service,
            "resolve_provider_auth_or_conflict",
            fake_auth,
        ),
        mock.patch.object(runtime, "require_crypto_box", lambda: object()),
    ]
    with _stack(patches):
        await auto_review._review_tool_call(
            mock.MagicMock(),
            "sess_1",
            call or _call(),
            account_id="acct_1",
            agent=surface or _surface(),
        )


class _stack:
    def __init__(self, patches: list[Any]) -> None:
        self._patches = patches

    def __enter__(self) -> None:
        for p in self._patches:
            p.start()

    def __exit__(self, *exc: Any) -> None:
        for p in self._patches:
            p.stop()


# ── verdict parsing ─────────────────────────────────────────────────────────


class TestParseVerdict:
    def test_allow(self) -> None:
        assert _parse_verdict('{"verdict": "allow", "reason": "routine read"}') == (
            "allow",
            "routine read",
        )

    def test_ask(self) -> None:
        assert _parse_verdict('{"verdict": "ask", "reason": "surprise recipient"}') == (
            "ask",
            "surprise recipient",
        )

    def test_fenced_json(self) -> None:
        parsed = _parse_verdict('```json\n{"verdict": "allow", "reason": "ok"}\n```')
        assert parsed == ("allow", "ok")

    def test_embedded_json(self) -> None:
        parsed = _parse_verdict('Sure! {"verdict": "ask", "reason": "risky"} there.')
        assert parsed == ("ask", "risky")

    def test_junk_is_none(self) -> None:
        assert _parse_verdict("I think this is fine") is None
        assert _parse_verdict("") is None
        assert _parse_verdict(None) is None
        assert _parse_verdict("[1, 2]") is None
        assert _parse_verdict('{"no_verdict": true}') is None

    def test_deny_shaped_coerces_to_ask(self) -> None:
        # Luna returns only allow|ask. A block/deny-shaped verdict is a real,
        # well-formed response clamped to the checker's actual authority.
        parsed = _parse_verdict('{"verdict": "deny", "reason": "bad"}')
        assert parsed is not None and parsed[0] == "ask"
        parsed = _parse_verdict('{"verdict": "block", "reason": "bad"}')
        assert parsed is not None and parsed[0] == "ask"

    def test_missing_reason_gets_default(self) -> None:
        parsed = _parse_verdict('{"verdict": "ask"}')
        assert parsed is not None and parsed[1]

    def test_long_reason_truncated(self) -> None:
        parsed = _parse_verdict('{"verdict": "ask", "reason": "' + "x" * 500 + '"}')
        assert parsed is not None and len(parsed[1]) <= 201


# ── prompt assembly ─────────────────────────────────────────────────────────


class TestBuildMessages:
    def test_args_fenced_and_scope_present(self) -> None:
        msgs = _build_messages(_surface("Accountant. Keeps the books."), _call(), ["[user] hi"])
        assert msgs[0]["role"] == "system"
        body = msgs[1]["content"]
        assert "<<<ARGS" in body and "ARGS>>>" in body
        assert "Accountant. Keeps the books." in body
        assert "server: gmail" in body and "tool: send_email" in body
        assert "[user] hi" in body

    def test_untrusted_framing_in_system(self) -> None:
        system = _build_messages(_surface(), _call(), [])[0]["content"]
        assert "cannot authorize" in system
        assert '"allow" | "ask"' in system

    def test_no_description_renders_placeholder(self) -> None:
        body = _build_messages(_surface(None), _call(), [])[1]["content"]
        assert "(none configured)" in body

    def test_args_truncated(self) -> None:
        call = _call()
        call["function"]["arguments"] = "x" * 10_000
        body = _build_messages(_surface(), call, [])[1]["content"]
        assert len(body) < 10_000


@pytest.mark.asyncio
async def test_recent_user_lines_labels_trigger_wakes() -> None:
    events = [
        SimpleNamespace(data={"role": "assistant", "content": "done"}),
        SimpleNamespace(
            data={
                "role": "user",
                "content": "check the mail",
                "metadata": {"trigger": {"id": "trg_1", "name": "morning-mail"}},
            }
        ),
        SimpleNamespace(data={"role": "user", "content": "hello", "metadata": {}}),
        SimpleNamespace(data={"role": "tool", "content": "result"}),
    ]

    async def read_events(pool: Any, sid: str, **kw: Any) -> list[Any]:
        return events

    with mock.patch.object(sessions_service, "read_events", read_events):
        lines = await _recent_user_lines(mock.MagicMock(), "s", account_id="a")
    assert lines == ["[user] hello", "[routine wake: morning-mail] check the mail"]


# ── orchestration ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_allow_confirms_with_source_and_wakes() -> None:
    rec = _Recorder()
    await _run_review(
        rec, model_results=[_response('{"verdict": "allow", "reason": "user asked"}')]
    )
    assert rec.confirms == [{"tool_call_id": "tc_1", "source": AUTO_REVIEW_SOURCE}]
    assert rec.wakes == ["auto_review"]
    assert rec.markers() == []  # allow never holds a card
    verdicts = rec.spans(AUTO_REVIEW_SPAN_EVENT)
    assert len(verdicts) == 1 and verdicts[0]["verdict"] == "allow"
    assert verdicts[0]["reason"] == "user asked"
    assert verdicts[0]["tool_call_id"] == "tc_1"
    assert verdicts[0]["model"] == "openai/responses/gpt-5.6-luna"
    assert isinstance(verdicts[0]["latency_ms"], int)
    # The real model call is metered: span pair + usage increment.
    assert len(rec.spans("model_request_end")) == 1
    assert rec.usage_increments and rec.usage_increments[0]["input_tokens"] == 100


@pytest.mark.asyncio
async def test_ask_holds_marker_with_reason_and_does_not_wake() -> None:
    rec = _Recorder()
    await _run_review(
        rec, model_results=[_response('{"verdict": "ask", "reason": "surprise recipient"}')]
    )
    assert rec.confirms == [] and rec.wakes == []
    markers = rec.markers()
    assert len(markers) == 1
    assert markers[0]["tool_call_id"] == "tc_1"
    assert markers[0]["kind"] == "mcp"
    assert markers[0]["reason"] == "surprise recipient"
    assert markers[0]["source"] == AUTO_REVIEW_SOURCE
    verdicts = rec.spans(AUTO_REVIEW_SPAN_EVENT)
    assert len(verdicts) == 1 and verdicts[0]["verdict"] == "ask"


@pytest.mark.asyncio
async def test_timeout_fails_closed_to_ask() -> None:
    rec = _Recorder()
    await _run_review(
        rec,
        model_results=["hang"],
        settings=_settings(auto_review_timeout_s=0.05),
    )
    markers = rec.markers()
    assert len(markers) == 1
    assert markers[0]["reason"] == CHECKER_UNAVAILABLE_REASON
    assert rec.confirms == [] and rec.wakes == []
    verdicts = rec.spans(AUTO_REVIEW_SPAN_EVENT)
    assert verdicts[0]["verdict"] == "ask"


@pytest.mark.asyncio
async def test_transient_failure_retries_once_then_succeeds() -> None:
    rec = _Recorder()
    await _run_review(
        rec,
        model_results=[
            RuntimeError("connection reset"),
            _response('{"verdict": "allow", "reason": "ok"}'),
        ],
    )
    assert rec.confirms and rec.spans(AUTO_REVIEW_SPAN_EVENT)[0]["verdict"] == "allow"


@pytest.mark.asyncio
async def test_junk_twice_fails_closed() -> None:
    rec = _Recorder()
    await _run_review(
        rec,
        model_results=[_response("not json"), _response("still not json")],
    )
    markers = rec.markers()
    assert len(markers) == 1 and markers[0]["reason"] == CHECKER_UNAVAILABLE_REASON


@pytest.mark.asyncio
async def test_two_failures_exhaust_the_single_retry() -> None:
    rec = _Recorder()
    await _run_review(
        rec,
        model_results=[RuntimeError("boom"), RuntimeError("boom again")],
    )
    assert rec.markers()[0]["reason"] == CHECKER_UNAVAILABLE_REASON
    assert rec.confirms == []


@pytest.mark.asyncio
async def test_provider_conflict_fails_closed_without_model_call() -> None:
    rec = _Recorder()
    await _run_review(
        rec,
        model_results=[],  # a model call would IndexError
        auth_result=(None, "cross-tenant key conflict"),
    )
    assert rec.markers()[0]["reason"] == CHECKER_UNAVAILABLE_REASON


@pytest.mark.asyncio
async def test_account_only_without_auth_fails_closed() -> None:
    rec = _Recorder()
    await _run_review(
        rec,
        model_results=[],
        settings=_settings(inference_credential_policy="account_only"),
        auth_result=(None, None),
    )
    assert rec.markers()[0]["reason"] == CHECKER_UNAVAILABLE_REASON


@pytest.mark.asyncio
async def test_existing_marker_is_not_duplicated() -> None:
    rec = _Recorder(marker_exists=True)
    await _run_review(rec, model_results=[_response('{"verdict": "ask", "reason": "risky"}')])
    assert rec.markers() == []  # idempotent against the sweep's writer
    assert rec.spans(AUTO_REVIEW_SPAN_EVENT)  # the verdict is still logged


@pytest.mark.asyncio
async def test_conflicted_confirm_is_dropped() -> None:
    rec = _Recorder(confirm_error=ConflictError("already resolved"))
    await _run_review(rec, model_results=[_response('{"verdict": "allow", "reason": "ok"}')])
    assert rec.wakes == []  # no wake after a dropped confirm
    assert rec.markers() == []


@pytest.mark.asyncio
async def test_interrupt_during_review_drops_the_allow() -> None:
    # Floor read at review start: None. Re-read at apply: seq 7 → an
    # interrupt landed mid-review; a checker allow is not fresh human intent.
    rec = _Recorder(interrupt_seqs=[None, 7])
    await _run_review(rec, model_results=[_response('{"verdict": "allow", "reason": "ok"}')])
    assert rec.confirms == [] and rec.wakes == []
    # The verdict is still logged; the call is left for the sweep backstop.
    assert rec.spans(AUTO_REVIEW_SPAN_EVENT)[0]["verdict"] == "allow"


# ── launcher ────────────────────────────────────────────────────────────────


def test_launch_registers_namespaced_keys_and_skips_idless() -> None:
    added: list[str] = []

    class _Reg:
        def add(self, sid: str, key: str, task: Any) -> None:
            added.append(key)
            task.cancel()  # don't actually run the review

        def remove(self, sid: str, key: str) -> None:
            pass

    async def main() -> None:
        with mock.patch.object(runtime, "require_inflight_tool_registry", lambda: _Reg()):
            launch_auto_review(
                mock.MagicMock(),
                "sess_1",
                [_call("tc_a"), {"function": {"name": "mcp__x__y"}}, _call("tc_b")],
                account_id="acct_1",
                agent=_surface(),
            )
        await asyncio.sleep(0)  # let the cancelled tasks unwind

    asyncio.run(main())
    assert added == [f"{REVIEW_KEY_PREFIX}tc_a", f"{REVIEW_KEY_PREFIX}tc_b"]
