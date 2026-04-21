"""``aios vault-credentials <verb>`` — operator CLI for credential CRUD.

Verbs: ``list`` (#108) and ``create`` (this PR).  ``create`` takes the
full ``VaultCredentialCreate`` body from a file (``--body-file <path>``
or ``-`` for stdin) rather than per-field flags — the schema branches
on ``auth_type`` with several ``SecretStr`` fields, and passing
secrets as shell args would leak them to shell history.  The
``kubectl create -f`` pattern sidesteps both concerns.

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
        description="Manage aios vault credentials.",
    )
    sub = parser.add_subparsers(dest="verb")

    lst = sub.add_parser("list", help="List credentials in a vault")
    lst.add_argument("--vault-id", required=True, help="Owning vault id")
    lst.add_argument("--limit", type=int, default=None, help="Max items per page")
    lst.add_argument(
        "--after", default=None, help="Pagination cursor (pass next_after from prior response)"
    )

    get = sub.add_parser("get", help="Show a single credential by id")
    get.add_argument("credential_id", help="Credential id")
    get.add_argument("--vault-id", required=True, help="Owning vault id")

    archive = sub.add_parser(
        "archive",
        help="Archive a credential (soft-delete, retained for audit)",
    )
    archive.add_argument("credential_id", help="Credential id")
    archive.add_argument("--vault-id", required=True, help="Owning vault id")

    delete = sub.add_parser(
        "delete",
        help="Hard-delete a credential (irreversible; prefer `archive` instead)",
    )
    delete.add_argument("credential_id", help="Credential id")
    delete.add_argument("--vault-id", required=True, help="Owning vault id")
    delete.add_argument(
        "--yes",
        action="store_true",
        help="Required to confirm hard-delete (no interactive prompt)",
    )

    update = sub.add_parser(
        "update",
        help="Update a credential (body from file; mcp_server_url + auth_type are immutable)",
    )
    update.add_argument("credential_id", help="Credential id")
    update.add_argument("--vault-id", required=True, help="Owning vault id")
    update.add_argument(
        "--body-file",
        required=True,
        help=(
            "Path to a JSON file containing the VaultCredentialUpdate body, "
            "or '-' to read from stdin.  Same secret-hygiene argument as "
            "`create`: tokens passed as shell args leak into history."
        ),
    )

    create = sub.add_parser("create", help="Create a new credential in a vault")
    create.add_argument("--vault-id", required=True, help="Owning vault id")
    create.add_argument(
        "--body-file",
        required=True,
        help=(
            "Path to a JSON file containing the full VaultCredentialCreate "
            "body, or '-' to read from stdin.  Using a file avoids leaking "
            "secret fields (tokens, client secrets) into shell history."
        ),
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
        return await _list(
            api_url, api_key, vault_id=args.vault_id, limit=args.limit, after=args.after
        )
    if args.verb == "get":
        return await _get(
            api_url, api_key, vault_id=args.vault_id, credential_id=args.credential_id
        )
    if args.verb == "archive":
        return await _archive(
            api_url, api_key, vault_id=args.vault_id, credential_id=args.credential_id
        )
    if args.verb == "delete":
        if not args.yes:
            print(
                f"{_PROG}: hard-delete is irreversible; pass --yes to confirm",
                file=sys.stderr,
            )
            return 2
        return await _delete(
            api_url, api_key, vault_id=args.vault_id, credential_id=args.credential_id
        )
    if args.verb == "update":
        try:
            body = _read_body(args.body_file)
        except CliError as err:
            print(str(err), file=sys.stderr)
            return 2
        return await _update(
            api_url,
            api_key,
            vault_id=args.vault_id,
            credential_id=args.credential_id,
            body=body,
        )
    if args.verb == "create":
        try:
            body = _read_body(args.body_file)
        except CliError as err:
            print(str(err), file=sys.stderr)
            return 2
        return await _create(api_url, api_key, vault_id=args.vault_id, body=body)
    parser.print_usage(sys.stderr)
    return 2


def _read_body(path: str) -> dict[str, Any]:
    """Read a JSON object from ``path``.  ``-`` reads stdin.

    Raises :class:`CliError` for IO failures, malformed JSON, or
    non-object top-level values.
    """
    if path == "-":
        raw = sys.stdin.read()
    else:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raise CliError(f"{_PROG}: cannot read --body-file {path!r}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(f"{_PROG}: --body-file contents are not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CliError(f"{_PROG}: --body-file must contain a JSON object")
    return parsed


async def _update(
    api_url: str,
    api_key: str,
    *,
    vault_id: str,
    credential_id: str,
    body: dict[str, Any],
) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials/{credential_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.put(url, headers=headers, json=body)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _delete(api_url: str, api_key: str, *, vault_id: str, credential_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials/{credential_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.delete(url, headers=headers)
    if response.status_code != 204:
        print_http_error(_PROG, response)
        return 2
    return 0


async def _archive(api_url: str, api_key: str, *, vault_id: str, credential_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials/{credential_id}/archive"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.post(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _get(api_url: str, api_key: str, *, vault_id: str, credential_id: str) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials/{credential_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0


async def _list(
    api_url: str, api_key: str, *, vault_id: str, limit: int | None, after: str | None
) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials"
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
    vault_id: str,
    body: dict[str, Any],
) -> int:
    url = f"{api_url.rstrip('/')}/v1/vaults/{vault_id}/credentials"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with async_client() as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code not in {200, 201}:
        print_http_error(_PROG, response)
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0
