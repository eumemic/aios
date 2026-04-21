"""``aios connections <verb>`` — operator CLI for connection CRUD.

Wraps ``POST``/``GET /v1/connections`` so the onboarding walkthrough
doesn't need a chain of curl invocations (#35 item 4).  Mirrors the
shape of ``aios tail``: reads ``AIOS_API_KEY`` + ``AIOS_API_URL`` /
``AIOS_API_HOST``+``AIOS_API_PORT`` from env and pipes through httpx.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx


def run(argv: list[str]) -> int:
    """Sync entry point for ``__main__``.  Tests use ``run_async`` directly."""
    return asyncio.run(run_async(argv))


async def run_async(argv: list[str]) -> int:
    """Parse ``argv`` and dispatch to a verb handler.

    ``argv`` is the slice *after* ``connections`` — e.g. for
    ``aios connections list`` this is ``["list"]``.
    """
    parser = argparse.ArgumentParser(
        prog="aios connections",
        description="Manage aios connections.",
    )
    sub = parser.add_subparsers(dest="verb")

    sub.add_parser("list", help="List connections")

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
        api_url, api_key = _require_env()
    except _CliError as err:
        print(str(err), file=sys.stderr)
        return 2

    if args.verb == "list":
        return await _list(api_url, api_key)
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


class _CliError(Exception):
    """Raised for user-visible config errors (missing env, etc.)."""


def _require_env() -> tuple[str, str]:
    api_key = os.environ.get("AIOS_API_KEY")
    if not api_key:
        raise _CliError("aios connections: AIOS_API_KEY is required")
    api_url = os.environ.get(
        "AIOS_API_URL",
        f"http://{os.environ.get('AIOS_API_HOST', '127.0.0.1')}"
        f":{os.environ.get('AIOS_API_PORT', '8080')}",
    )
    return api_url, api_key


async def _list(api_url: str, api_key: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        print(
            f"aios connections: HTTP {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 2
    body: dict[str, Any] = response.json()
    print(json.dumps(body.get("data", []), indent=2))
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code not in {200, 201}:
        print(
            f"aios connections: HTTP {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0
