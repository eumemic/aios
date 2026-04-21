"""``aios bindings <verb>`` — operator CLI for channel-binding CRUD.

Wraps ``POST``/``GET /v1/channel-bindings`` so the onboarding
walkthrough doesn't need a chain of curl invocations (#35 item 4).
Mirrors :mod:`aios.cli.connections`: reads ``AIOS_API_KEY`` +
``AIOS_API_URL`` / ``AIOS_API_HOST``+``AIOS_API_PORT`` from env and
pipes through httpx.
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

    ``argv`` is the slice *after* ``bindings`` — e.g. for
    ``aios bindings list`` this is ``["list"]``.
    """
    parser = argparse.ArgumentParser(
        prog="aios bindings",
        description="Manage aios channel bindings (address → session mappings).",
    )
    sub = parser.add_subparsers(dest="verb")

    lst = sub.add_parser("list", help="List channel bindings")
    lst.add_argument(
        "--session-id",
        default=None,
        help="Filter to bindings for this session id",
    )

    create = sub.add_parser("create", help="Create a new channel binding")
    create.add_argument(
        "--address",
        required=True,
        help="Full channel address (e.g. signal/+15550001/group/abc)",
    )
    create.add_argument(
        "--session-id",
        required=True,
        help="Session id to bind the address to",
    )

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
        return await _list(api_url, api_key, session_id=args.session_id)
    if args.verb == "create":
        return await _create(
            api_url,
            api_key,
            address=args.address,
            session_id=args.session_id,
        )
    parser.print_usage(sys.stderr)
    return 2


class _CliError(Exception):
    """Raised for user-visible config errors (missing env, etc.)."""


def _require_env() -> tuple[str, str]:
    api_key = os.environ.get("AIOS_API_KEY")
    if not api_key:
        raise _CliError("aios bindings: AIOS_API_KEY is required")
    api_url = os.environ.get(
        "AIOS_API_URL",
        f"http://{os.environ.get('AIOS_API_HOST', '127.0.0.1')}"
        f":{os.environ.get('AIOS_API_PORT', '8080')}",
    )
    return api_url, api_key


async def _list(api_url: str, api_key: str, *, session_id: str | None) -> int:
    url = f"{api_url.rstrip('/')}/v1/channel-bindings"
    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, str] = {}
    if session_id is not None:
        params["session_id"] = session_id
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(
            f"aios bindings: HTTP {response.status_code}: {response.text}",
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
    address: str,
    session_id: str,
) -> int:
    url = f"{api_url.rstrip('/')}/v1/channel-bindings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"address": address, "session_id": session_id}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code not in {200, 201}:
        print(
            f"aios bindings: HTTP {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(response.json(), indent=2))
    return 0
