"""``aios vaults <verb>`` — operator CLI for vault CRUD.

Wraps ``POST``/``GET /v1/vaults`` (#35 item 4).  The ``metadata`` field
on ``VaultCreate`` is a free-form JSON object; CLI accepts it as
``--metadata-json`` (optional; default ``{}``) — same shape as
``aios rules --session-params-json``.

Credential management (``/v1/vaults/<id>/credentials``) is deliberately
NOT surfaced here — the credential payload is ``auth_type``-dependent
with several ``SecretStr`` branches and warrants its own
``aios vaults credentials <verb>`` subcommand group in a later PR.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from aios.cli._http import CliError, async_client, print_http_error, require_env

_PROG = "aios vaults"


def run(argv: list[str]) -> int:
    """Sync entry point for ``__main__``.  Tests use ``run_async`` directly."""
    return asyncio.run(run_async(argv))


async def run_async(argv: list[str]) -> int:
    """Parse ``argv`` and dispatch to a verb handler.

    ``argv`` is the slice *after* ``vaults``.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Manage aios vaults.",
    )
    sub = parser.add_subparsers(dest="verb")

    sub.add_parser("list", help="List vaults")

    create = sub.add_parser("create", help="Create a new vault")
    create.add_argument(
        "--display-name",
        required=True,
        help="Human-readable name for the vault (1-128 chars)",
    )
    create.add_argument(
        "--metadata-json",
        default=None,
        help="JSON object of vault metadata (default: empty object)",
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
        return await _list(api_url, api_key)
    if args.verb == "create":
        try:
            metadata = _parse_metadata(args.metadata_json)
        except CliError as err:
            print(str(err), file=sys.stderr)
            return 2
        return await _create(
            api_url,
            api_key,
            display_name=args.display_name,
            metadata=metadata,
        )
    parser.print_usage(sys.stderr)
    return 2


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(f"{_PROG}: --metadata-json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CliError(f"{_PROG}: --metadata-json must be a JSON object")
    return parsed


async def _list(api_url: str, api_key: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults"
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
    display_name: str,
    metadata: dict[str, Any],
) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"display_name": display_name, "metadata": metadata}
    async with async_client() as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code not in {200, 201}:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0
