"""CLI entrypoint: ``aios <subcommand>`` / ``python -m aios <subcommand>``.

The subcommand surface (operator commands + client commands + chat REPL) is
defined in :mod:`aios.cli.app`. This module is a thin delegator so both
``[project.scripts]`` and ``python -m aios`` reach the same typer app.
"""

from __future__ import annotations


def main() -> int:
    from aios.cli.app import app

    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
