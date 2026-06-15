"""Unit tests for the dev-pipeline ``_is_transient`` retry classifier (#1139).

``_is_transient`` lives inside the workflow *script source* (``_BODY``), so it is not
importable as a module attribute. We build the production script and ``exec`` it in a
fresh namespace — the body imports only ``json``/``re`` and references header constants,
both satisfied by ``build_dev_pipeline_script`` — then pull the function out of the
namespace and exercise it directly. The classifier is pure (no LLM, no tool, no time).

The bug (#1139, dogfood 2026-06-14): the old classifier treated ANY ``{"error": ...}``
tool result as transient, so a deterministic broker route-allowlist rejection
("does not match any enabled route", a 405-equivalent) burned three pointless retries on
every call. Only genuine transients — 5xx, request timeouts, and HTTP transport errors —
must retry; every broker gate rejection (route mismatch / method-not-allowed / unknown
server / SSRF block / path rejection) is terminal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script


@pytest.fixture(scope="module")
def is_transient() -> Callable[[Any], bool]:
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    namespace: dict[str, Any] = {}
    exec(compile(src, "dev_pipeline_script", "exec"), namespace)
    fn: Callable[[Any], bool] = namespace["_is_transient"]
    return fn


# ─── genuine transients: retry ────────────────────────────────────────────────


def test_5xx_status_is_transient(is_transient: Callable[[Any], bool]) -> None:
    for status in (500, 502, 503, 599):
        assert is_transient({"status": status}) is True


def test_request_timeout_error_is_transient(is_transient: Callable[[Any], bool]) -> None:
    # http_request surfaces a timeout as {"error": "Request timed out: ..."}.
    assert is_transient({"error": "Request timed out: GET https://api.github.com/x"}) is True


def test_transport_error_is_transient(is_transient: Callable[[Any], bool]) -> None:
    # A connection reset / DNS failure surfaces as an HTTP transport error envelope.
    assert is_transient({"error": "HTTP transport error: ConnectError: connection reset"}) is True


def test_non_dict_result_is_transient(is_transient: Callable[[Any], bool]) -> None:
    # A malformed (non-dict) tool return is treated as transient.
    assert is_transient("boom") is True
    assert is_transient(None) is True


# ─── deterministic broker/route rejections: terminal (the #1139 fix) ──────────


def test_route_mismatch_error_is_terminal(is_transient: Callable[[Any], bool]) -> None:
    # The exact broker route-allowlist rejection from http_request.py — a 405-equivalent.
    resp = {
        "error": (
            "DELETE '/repos/o/r/issues/1/labels/x' does not match any enabled route on "
            "http_server 'github' — the path may be unlisted, or the route's allowed "
            "methods may not include this verb"
        )
    }
    assert is_transient(resp) is False


def test_unknown_server_ref_error_is_terminal(is_transient: Callable[[Any], bool]) -> None:
    resp = {"error": "unknown server_ref 'github'; not declared on http_servers"}
    assert is_transient(resp) is False


def test_ssrf_block_error_is_terminal(is_transient: Callable[[Any], bool]) -> None:
    resp = {"error": "Blocked: URL targets a private/internal address: http://10.0.0.1/x"}
    assert is_transient(resp) is False


def test_path_rejection_error_is_terminal(is_transient: Callable[[Any], bool]) -> None:
    resp = {"error": "path may not contain a query string"}
    assert is_transient(resp) is False


# ─── HTTP status results: 4xx/2xx are the caller's branch, never retried ───────


def test_4xx_status_is_terminal(is_transient: Callable[[Any], bool]) -> None:
    for status in (400, 401, 403, 404, 405, 422):
        assert is_transient({"status": status}) is False


def test_2xx_status_is_terminal(is_transient: Callable[[Any], bool]) -> None:
    for status in (200, 201, 204):
        assert is_transient({"status": status}) is False
