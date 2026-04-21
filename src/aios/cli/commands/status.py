"""``aios status`` — check reachability + auth of the configured API."""

from __future__ import annotations

import sys

import typer

from aios.cli.client import AiosApiError
from aios.cli.output import cyan, green, print_json, red, yellow
from aios.cli.runtime import get_state, run_or_die


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

    def _run_text(state) -> int:  # type: ignore[no-untyped-def]
        sys.stdout.write(f"url:     {cyan(state.base_url)}\n")
        sys.stdout.write(
            f"api key: {'set' if state.api_key else yellow('not set (no auth header sent)')}\n"
        )
        client = state.client()
        try:
            health = client.request("GET", "/health")
            sys.stdout.write(f"health:  {green('ok')} ({health})\n")
        except AiosApiError as exc:
            sys.stdout.write(f"health:  {red('fail')} ({exc.message})\n")
            return 1
        if not state.api_key:
            return 0
        try:
            client.request("GET", "/v1/agents", params={"limit": 1})
            sys.stdout.write(f"auth:    {green('ok')}\n")
        except AiosApiError as exc:
            if exc.status_code == 401:
                sys.stdout.write(f"auth:    {red('unauthorized')} — check AIOS_API_KEY\n")
                return 2
            sys.stdout.write(f"auth:    {red('fail')} ({exc.error_type}: {exc.message})\n")
            return 1
        return 0

    def _run_json(state) -> int:  # type: ignore[no-untyped-def]
        client = state.client()
        payload: dict[str, object] = {"url": state.base_url, "api_key_set": bool(state.api_key)}
        try:
            payload["health"] = client.request("GET", "/health")
        except AiosApiError as exc:
            payload["health_error"] = {"type": exc.error_type, "message": exc.message}
            print_json(payload)
            return 1
        if not state.api_key:
            print_json(payload)
            return 0
        try:
            client.request("GET", "/v1/agents", params={"limit": 1})
            payload["auth"] = "ok"
        except AiosApiError as exc:
            payload["auth"] = "unauthorized" if exc.status_code == 401 else "fail"
            payload["auth_error"] = {"type": exc.error_type, "message": exc.message}
            print_json(payload)
            return 2 if exc.status_code == 401 else 1
        print_json(payload)
        return 0
