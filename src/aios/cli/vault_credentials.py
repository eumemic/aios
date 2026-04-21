"""``aios vault-credentials <verb>`` — operator CLI for credential inspection.

Scope for this module: ``list`` only.  Credential *creation* has an
``auth_type``-dependent schema branch (``mcp_oauth`` vs
``static_bearer``) with several ``SecretStr`` fields whose CLI flag
design deserves its own PR.

Uses :mod:`aios.cli._http` for env + HTTP + error-format plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from aios.cli._http import CliError, async_client, print_http_error, require_env

_PROG = "aios vault-credentials"


def run(argv: list[str]) -> int:
    """Sync entry point for ``__main__``.  Tests use ``run_async`` directly."""
    return asyncio.run(run_async(argv))


async def run_async(argv: list[str]) -> int:
    """Parse ``argv`` and dispatch to a verb handler."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Inspect aios vault credentials.",
    )
    sub = parser.add_subparsers(dest="verb")

    lst = sub.add_parser("list", help="List credentials in a vault")
    lst.add_argument("--vault-id", required=True, help="Owning vault id")

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    if args.verb is None:
        parser.print_usage(sys.stderr)
        return 2

    try:
        api_url, api_key = require_env(_PROG)
    except CliError as err:
        print(str(err), file=sys.stderr)
        return 2

    if args.verb == "list":
        return await _list(api_url, api_key, vault_id=args.vault_id)
    parser.print_usage(sys.stderr)
    return 2


async def _list(api_url: str, api_key: str, *, vault_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    body: dict[str, Any] = response.json()
    print(json.dumps(body.get("data", []), indent=2))
    return 0
