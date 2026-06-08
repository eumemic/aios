"""Regression test for ``--all`` walks of path-param'd list endpoints.

The opaque ``next_cursor`` re-encodes the query filters (so the client needn't
resend them, and the server 422s if it does), but a URL **path** segment like
``/v1/vaults/{vault_id}/credentials`` is structural — never in the cursor — so it
must be resent on every page. ``render_paginated`` carries those via ``path_params``
(not ``**filters``). A command that mis-wired its id as a filter would drop it on
page 2, where the SDK call is ``fn(client=..., cursor=...)`` with no id, raising
``TypeError`` on the missing required argument.

This drives every path-param'd list command across a real two-page boundary and
asserts the id survives onto page 2 — the contract for all of them at once.
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()

# (argv, page-2 URL path that must still carry the id). The id lives in the path,
# so the page-2 request — made with only ``?cursor=`` — must still hit this URL.
_PATH_PARAM_LIST_COMMANDS = [
    pytest.param(
        ["agents", "versions", "agt_1", "--all"], "/v1/agents/agt_1/versions", id="agents"
    ),
    pytest.param(
        ["skills", "versions", "skl_1", "--all"], "/v1/skills/skl_1/versions", id="skills"
    ),
    pytest.param(
        ["vaults", "credentials", "list", "vlt_1", "--all"],
        "/v1/vaults/vlt_1/credentials",
        id="vaults",
    ),
    pytest.param(["runs", "events", "wfr_1", "--all"], "/v1/runs/wfr_1/events", id="runs"),
]


@pytest.mark.parametrize("argv, page2_path", _PATH_PARAM_LIST_COMMANDS)
def test_all_walk_preserves_url_path_param(mocked_cli, argv, page2_path):
    # Page 1 advertises a next page; page 2 ends the walk. Empty ``data`` keeps the
    # SDK parser from needing item fixtures — we only care about request shape.
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": [], "has_more": True, "next_cursor": "CUR1"})
    )
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": [], "has_more": False, "next_cursor": None})
    )

    result = runner.invoke(app, argv)

    # Buggy call sites (id passed as a query filter) raise TypeError on page 2.
    assert result.exit_code == 0, result.output
    # ``captured`` holds the last request — page 2 — which carried only the cursor,
    # so the id can only have survived via the URL path.
    assert mocked_cli.captured.path == page2_path
    assert mocked_cli.captured.query.get("cursor") == ["CUR1"]
