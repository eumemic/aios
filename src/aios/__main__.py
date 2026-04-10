"""CLI entrypoint: ``python -m aios <subcommand>``.

Phase 1 wires up ``api`` (uvicorn) and ``migrate`` (alembic). The ``worker``
subcommand is wired up in Phase 2 when the procrastinate worker lands.
"""

from __future__ import annotations

import subprocess
import sys


def _run_api() -> int:
    import uvicorn

    from aios.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "aios.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # we configure structlog ourselves
    )
    return 0


def _run_migrate() -> int:
    return subprocess.call(["alembic", "upgrade", "head"])


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: aios <api|worker|migrate>", file=sys.stderr)
        return 2

    cmd = sys.argv[1]
    match cmd:
        case "api":
            return _run_api()
        case "migrate":
            return _run_migrate()
        case "worker":
            print(
                "aios worker: not yet wired up (phase 2 implements the worker)",
                file=sys.stderr,
            )
            return 1
        case _:
            print(f"aios: unknown subcommand {cmd!r}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
