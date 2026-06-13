"""Tests for the ``http_servers`` system-prompt allowlist block (#828).

The block renders each enabled route's allowed HTTP methods so the model
knows which verbs it may use. ``methods=None`` renders as ``ANY`` (all
methods); a scoped list renders the verbs comma-joined and sorted.
"""

from __future__ import annotations

from aios.harness.step_context import _build_http_servers_block
from aios.models.agents import HttpRouteSpec, HttpServerSpec


def test_method_scoped_route_renders_verb_prefix() -> None:
    block = _build_http_servers_block(
        [
            HttpServerSpec(
                name="api",
                base_url="https://api",
                routes=[
                    HttpRouteSpec(path_pattern="/repos/**", methods=["GET", "POST"]),
                ],
            )
        ]
    )
    assert "GET,POST /repos/**" in block


def test_unscoped_route_renders_any() -> None:
    block = _build_http_servers_block(
        [
            HttpServerSpec(
                name="api",
                base_url="https://api",
                routes=[HttpRouteSpec(path_pattern="/x")],
            )
        ]
    )
    assert "ANY /x" in block


def test_route_description_preserved_after_verb() -> None:
    block = _build_http_servers_block(
        [
            HttpServerSpec(
                name="api",
                base_url="https://api",
                routes=[
                    HttpRouteSpec(path_pattern="/y", methods=["GET"], description="read a thing")
                ],
            )
        ]
    )
    assert "GET /y — read a thing" in block
