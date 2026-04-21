"""``aios connections <verb>`` — operator CLI for connection CRUD.

Wraps ``POST``/``GET /v1/connections`` so the onboarding walkthrough
doesn't need a chain of curl invocations (#35 item 4).  Env handling,
``httpx`` setup, and HTTP-error formatting live in
:mod:`aios.cli._http`; this module stays focused on argv shape and the
two verb handlers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from aios.cli._http import CliError, async_client, print_http_error, require_env

_PROG = "aios connections"


def run(argv: list[str]) -> int:
    """Sync entry point for ``__main__``.  Tests use ``run_async`` directly."""
    return asyncio.run(run_async(argv))


async def run_async(argv: list[str]) -> int:
    """Parse ``argv`` and dispatch to a verb handler.

    ``argv`` is the slice *after* ``connections`` — e.g. for
    ``aios connections list`` this is ``["list"]``.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Manage aios connections.",
    )
    sub = parser.add_subparsers(dest="verb")

    lst = sub.add_parser("list", help="List connections")
    lst.add_argument("--limit", type=int, default=None, help="Max items per page")
    lst.add_argument(
        "--after", default=None, help="Pagination cursor (pass next_after from prior response)"
    )

    get = sub.add_parser("get", help="Show a single connection by id")
    get.add_argument("connection_id", help="Connection id")

    archive = sub.add_parser(
        "archive",
        help="Archive a connection (soft-delete, retained for audit)",
    )
    archive.add_argument("connection_id", help="Connection id")

    create = sub.add_parser("create", help="Create a new connection")
    create.add_argument("--connector", required=True, help="Connector type (e.g. signal)")
    create.add_argument("--account", required=True, help="Account identifier (e.g. bot uuid)")
    create.add_argument("--mcp-url", required=True, help="MCP server URL")
    create.add_argument("--vault-id", required=True, help="Vault id with the MCP credential")

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
        return await _list(api_url, api_key, limit=args.limit, after=args.after)
    if args.verb == "get":
        return await _get(api_url, api_key, connection_id=args.connection_id)
    if args.verb == "archive":
        return await _archive(api_url, api_key, connection_id=args.connection_id)
    if args.verb == "create":
        return await _create(
            api_url,
            api_key,
            connector=args.connector,
            account=args.account,
            mcp_url=args.mcp_url,
            vault_id=args.vault_id,
        )
    parser.print_usage(sys.stderr)
    return 2


async def _archive(api_url: str, api_key: str, *, connection_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections/{connection_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.delete(url, headers=headers)
    if response.status_code != 204:
        print_http_error(_PROG, response)
        return 2
    return 0


async def _get(api_url: str, api_key: str, *, connection_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections/{connection_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _list(api_url: str, api_key: str, *, limit: int | None, after: str | None) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections"
    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if after is not None:
        params["after"] = after
    async with async_client() as client:
        response = await client.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _create(
    api_url: str,
    api_key: str,
    *,
    connector: str,
    account: str,
    mcp_url: str,
    vault_id: str,
) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "connector": connector,
        "account": account,
        "mcp_url": mcp_url,
        "vault_id": vault_id,
    }
    async with async_client() as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code not in {200, 201}:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0
