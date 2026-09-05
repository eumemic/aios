"""Regression guard for the ``aios envs list`` rendered column set.

The ``envs`` CLI surface previously had no unit tests, which is how a
``_COLS`` tuple referencing a non-existent ``updated_at`` field (while
omitting the real ``created_at``) survived. This test pins the rendered
columns to the ``Environment`` resource shape so the footgun cannot regress.
"""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()

_TS1 = "2024-01-01T00:00:00+00:00"
_TS2 = "2024-06-15T12:30:00+00:00"


def test_envs_list_renders_created_at_and_omits_updated_at(mocked_cli):
    """``aios envs list`` must surface ``created_at`` and must not render a
    ``UPDATED_AT`` column — the ``Environment`` resource has no ``updated_at``
    field at any layer of the stack, while ``created_at`` is present on every
    row. The list endpoint excludes archived rows by default, so every
    returned row carries a real ``created_at`` and a null ``archived_at``.
    """
    rows: list[dict[str, object]] = [
        {"id": "env_1", "name": "dev", "config": {}, "created_at": _TS1, "archived_at": None},
        {"id": "env_2", "name": "staging", "config": {}, "created_at": _TS2, "archived_at": None},
    ]
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": rows, "has_more": False, "next_cursor": None})
    )
    result = runner.invoke(app, ["envs", "list"])
    assert result.exit_code == 0, result.output

    out = result.output.upper()
    assert "CREATED_AT" in out
    assert "UPDATED_AT" not in out
    assert "ARCHIVED_AT" in out

    # The real per-row created_at timestamps render (the prior bug left them
    # hidden behind a non-existent updated_at column).
    assert "2024-01-01" in result.output
    assert "2024-06-15" in result.output
    assert "env_1" in result.output
    assert "staging" in result.output
