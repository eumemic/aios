"""Top-level ``aios tail <session_id>`` — structured SSE viewer.

Alias for ``aios sessions tail`` kept at the top level because operators
reach for it constantly for debugging live turns. Shares the formatter
in :mod:`aios.cli.tail_format` with the canonical subcommand.
"""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands.sessions import tail_session
from aios.cli.runtime import run_or_die


def register(app: typer.Typer) -> None:
    @app.command(
        "tail",
        help="Structured one-line real-time viewer for a session (top-level alias).",
    )
    def tail(
        ctx: typer.Context,
        session_id: str,
        after_seq: Annotated[int, typer.Option("--after-seq", min=0)] = 0,
    ) -> None:
        def _run() -> None:
            tail_session(ctx, session_id, after_seq=after_seq)

        run_or_die(_run)
