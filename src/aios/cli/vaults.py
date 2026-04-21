"""``aios vaults <verb>`` — operator CLI for vault CRUD.

Wraps ``POST``/``GET /v1/vaults`` (#35 item 4).  The ``metadata`` field
on ``VaultCreate`` is a free-form JSON object; CLI accepts it as
``--metadata-json`` (optional; default ``{}``) — same shape as
``aios rules --session-params-json``.

Credential management (``/v1/vaults/<id>/credentials``) lives in its
own top-level sibling — :mod:`aios.cli.vault_credentials` — matching
the flat one-group-per-top-level-subcommand shape the other CLI
modules follow.
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

    lst = sub.add_parser("list", help="List vaults")
    lst.add_argument("--limit", type=int, default=None, help="Max items per page")
    lst.add_argument(
        "--after", default=None, help="Pagination cursor (pass next_after from prior response)"
    )

    get = sub.add_parser("get", help="Show a single vault by id")
    get.add_argument("vault_id", help="Vault id")

    archive = sub.add_parser(
        "archive",
        help="Archive a vault (soft-delete, retained for audit)",
    )
    archive.add_argument("vault_id", help="Vault id")

    delete = sub.add_parser(
        "delete",
        help="Hard-delete a vault (irreversible; prefer `archive` instead)",
    )
    delete.add_argument("vault_id", help="Vault id")
    delete.add_argument(
        "--yes",
        action="store_true",
        help="Required to confirm hard-delete (no interactive prompt)",
    )

    update = sub.add_parser("update", help="Update a vault's display name or metadata")
    update.add_argument("vault_id", help="Vault id")
    update.add_argument("--display-name", default=None, help="New display name")
    update.add_argument(
        "--metadata-json",
        default=None,
        help="New metadata (JSON object, replaces existing)",
    )

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
        return await _list(api_url, api_key, limit=args.limit, after=args.after)
    if args.verb == "get":
        return await _get(api_url, api_key, vault_id=args.vault_id)
    if args.verb == "archive":
        return await _archive(api_url, api_key, vault_id=args.vault_id)
    if args.verb == "delete":
        if not args.yes:
            print(
                f"{_PROG}: hard-delete is irreversible; pass --yes to confirm",
                file=sys.stderr,
            )
            return 2
        return await _delete(api_url, api_key, vault_id=args.vault_id)
    if args.verb == "update":
        metadata: dict[str, Any] | None = None
        if args.metadata_json is not None:
            try:
                metadata = _parse_metadata(args.metadata_json)
            except CliError as err:
                print(str(err), file=sys.stderr)
                return 2
        return await _update(
            api_url,
            api_key,
            vault_id=args.vault_id,
            display_name=args.display_name,
            metadata=metadata,
        )
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


async def _update(
    api_url: str,
    api_key: str,
    *,
    vault_id: str,
    display_name: str | None,
    metadata: dict[str, Any] | None,
) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    body: dict[str, Any] = {}
    if display_name is not None:
        body["display_name"] = display_name
    if metadata is not None:
        body["metadata"] = metadata
    async with async_client() as client:
        response = await client.put(url, headers=headers, json=body)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _delete(api_url: str, api_key: str, *, vault_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.delete(url, headers=headers)
    if response.status_code != 204:
        print_http_error(_PROG, response)
        return 2
    return 0


async def _archive(api_url: str, api_key: str, *, vault_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/archive"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.post(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _get(api_url: str, api_key: str, *, vault_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _list(api_url: str, api_key: str, *, limit: int | None, after: str | None) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults"
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
