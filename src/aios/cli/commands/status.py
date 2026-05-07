"""``aios status`` — check reachability + auth of the configured API."""

from __future__ import annotations

import json
import sys

import httpx
import typer

from aios.cli.output import cyan, green, print_json, red, yellow
from aios.cli.runtime import CliState, get_state, run_or_die
from aios.sdk._generated.api.agents import list_agents
from aios.sdk._generated.api.default import get_health


def register(app: typer.Typer) -> None:
    @app.command(
        "status",
        help="Print the configured API URL, reachability, and auth status.",
    )
    def status(ctx: typer.Context) -> None:
        state = get_state(ctx)

        def _run() -> int:
            if state.output_format == "json":
                return _run_json(state)
            return _run_text(state)

        run_or_die(_run)


def _check_health(state: CliState) -> tuple[dict[str, str] | None, str, str | None]:
    """Probe ``/health``. Returns ``(payload, kind, message)``.

    ``kind`` is ``"ok"``, ``"non_json_response"`` (server returned non-JSON — likely
    not aios), ``"connection"``, or ``"http"``. ``payload`` is non-None
    only when ``kind == "ok"``; ``message`` is non-None on errors.
    """
    client = state.sdk_client()
    try:
        response = get_health.sync_detailed(client=client)
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return None, "connection", f"connection error: {exc}"
    except json.JSONDecodeError:
        return None, "non_json_response", "server returned non-JSON response (not aios?)"
    if response.status_code != 200:
        return None, "http", f"HTTP {response.status_code}"
    return json.loads(response.content), "ok", None


def _check_auth(state: CliState) -> tuple[str, int]:
    """Probe ``/v1/agents`` to verify the API key. Returns ``(kind, status_code)``.

    ``kind`` is ``"ok"``, ``"unauthorized"`` (401), or ``"fail"``.
    """
    client = state.sdk_client()
    response = list_agents.sync_detailed(client=client, limit=1)
    if response.status_code == 200:
        return "ok", 200
    if response.status_code == 401:
        return "unauthorized", 401
    return "fail", int(response.status_code)


def _run_text(state: CliState) -> int:
    sys.stdout.write(f"url:     {cyan(state.base_url)}\n")
    sys.stdout.write(
        f"api key: {'set' if state.api_key else yellow('not set (no auth header sent)')}\n"
    )
    payload, kind, message = _check_health(state)
    if kind != "ok":
        sys.stdout.write(f"health:  {red('fail')} ({message})\n")
        return 1
    sys.stdout.write(f"health:  {green('ok')} ({payload})\n")
    if not state.api_key:
        return 0
    auth_kind, status_code = _check_auth(state)
    if auth_kind == "ok":
        sys.stdout.write(f"auth:    {green('ok')}\n")
        return 0
    if auth_kind == "unauthorized":
        sys.stdout.write(f"auth:    {red('unauthorized')} — check AIOS_API_KEY\n")
        return 2
    sys.stdout.write(f"auth:    {red('fail')} (HTTP {status_code})\n")
    return 1


def _run_json(state: CliState) -> int:
    payload: dict[str, object] = {"url": state.base_url, "api_key_set": bool(state.api_key)}
    health, kind, message = _check_health(state)
    if kind != "ok":
        payload["health_error"] = {"type": kind, "message": message}
        print_json(payload)
        return 1
    payload["health"] = health
    if not state.api_key:
        print_json(payload)
        return 0
    auth_kind, status_code = _check_auth(state)
    payload["auth"] = auth_kind
    if auth_kind != "ok":
        payload["auth_error"] = {"type": auth_kind, "status": status_code}
    print_json(payload)
    return 0 if auth_kind == "ok" else (2 if auth_kind == "unauthorized" else 1)
