"""``aios rules <verb>`` — operator CLI for routing-rule CRUD.

Routing rules are per-connection (``/v1/connections/<id>/routing-rules``)
so both verbs require ``--connection-id``.  The nested ``SessionParams``
block is accepted as a JSON string via ``--session-params-json``; absent
flag means the rule is created with an empty ``SessionParams``.

Env handling, ``httpx`` setup, and HTTP-error formatting live in
:mod:`aios.cli._http`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from aios.cli._http import CliError, async_client, print_http_error, require_env

_PROG = "aios rules"


def run(argv: list[str]) -> int:
    """Sync entry point for ``__main__``.  Tests use ``run_async`` directly."""
    return asyncio.run(run_async(argv))


async def run_async(argv: list[str]) -> int:
    """Parse ``argv`` and dispatch to a verb handler.

    ``argv`` is the slice *after* ``rules``.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Manage aios routing rules (per-connection).",
    )
    sub = parser.add_subparsers(dest="verb")

    lst = sub.add_parser("list", help="List routing rules for a connection")
    lst.add_argument("--connection-id", required=True, help="Owning connection id")

    create = sub.add_parser("create", help="Create a new routing rule")
    create.add_argument("--connection-id", required=True, help="Owning connection id")
    create.add_argument(
        "--prefix",
        required=True,
        help='Channel-path prefix ("" = per-connection catch-all)',
    )
    create.add_argument(
        "--target",
        required=True,
        help='Route target, e.g. "agent:claude-sonnet-4" or "session:sess_01"',
    )
    create.add_argument(
        "--session-params-json",
        default=None,
        help="JSON object for nested SessionParams (default: empty object)",
    )

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
        return await _list(api_url, api_key, connection_id=args.connection_id)
    if args.verb == "create":
        try:
            session_params = _parse_session_params(args.session_params_json)
        except CliError as err:
            print(str(err), file=sys.stderr)
            return 2
        return await _create(
            api_url,
            api_key,
            connection_id=args.connection_id,
            prefix=args.prefix,
            target=args.target,
            session_params=session_params,
        )
    parser.print_usage(sys.stderr)
    return 2


def _parse_session_params(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(f"{_PROG}: --session-params-json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CliError(f"{_PROG}: --session-params-json must be a JSON object")
    return parsed


async def _list(api_url: str, api_key: str, *, connection_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections/{connection_id}/routing-rules"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    body: dict[str, Any] = response.json()
    print(json.dumps(body.get("data", []), indent=2))
    return 0


async def _create(
    api_url: str,
    api_key: str,
    *,
    connection_id: str,
    prefix: str,
    target: str,
    session_params: dict[str, Any],
) -> int:
    url = f"{api_url.rstrip('/')}/v1/connections/{connection_id}/routing-rules"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"prefix": prefix, "target": target, "session_params": session_params}
    async with async_client() as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code not in {200, 201}:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0
