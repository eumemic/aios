"""``aios signal ...`` — signal-cli management operations via the aios API (#348).

Three commands cover the SSH-required gap:

* ``aios signal register +1...`` — initiate registration; prints the
  signalcaptcha URL and the exact retry command if captcha is required.
* ``aios signal verify +1... 123456`` — submit the SMS / voice code.
* ``aios signal profile +1... --given-name=Alice`` — update profile metadata.

Each command is a thin layer over the generated SDK operation; the
operator-facing route blocks until the connector resolves the call,
so the CLI experiences one HTTP round-trip per step.
"""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands._shared import render_single, unwrap
from aios.cli.output import print_error, print_note
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.connectors import (
    post_connector_signal_profile,
    post_connector_signal_register,
    post_connector_signal_verify,
)
from aios_sdk._generated.models.signal_profile_request import SignalProfileRequest
from aios_sdk._generated.models.signal_register_request import SignalRegisterRequest
from aios_sdk._generated.models.signal_verify_request import SignalVerifyRequest

app = typer.Typer(name="signal", help="Signal-cli management operations.", no_args_is_help=True)


@app.command("register")
def register(
    ctx: typer.Context,
    phone: Annotated[str, typer.Argument(help="E.164 phone, e.g. +15551234567.")],
    captcha: Annotated[
        str | None,
        typer.Option(
            "--captcha",
            help="signalcaptcha:// token from a prior captcha-required response.",
        ),
    ] = None,
    voice: Annotated[
        bool,
        typer.Option(
            "--voice",
            help="Use voice call instead of SMS for the verification code.",
        ),
    ] = False,
) -> None:
    """Initiate signal-cli registration for a phone number.

    If signal demands a captcha, prints the URL and the retry command —
    solve it in a browser, copy the resulting token, and re-run with
    ``--captcha=<token>``.  On success the user receives a 6-digit code
    via SMS (or voice with ``--voice``); submit it via ``aios signal verify``.
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = SignalRegisterRequest(account=phone, captcha=captcha, voice=voice)
            result = unwrap(post_connector_signal_register.sync_detailed(client=client, body=body))
        payload = result.to_dict()
        if payload.get("status") == "captcha_required":
            url = payload.get("captcha_url", "")
            print_error(
                f"signal requires a captcha — solve it at:\n  {url}\n"
                f"then retry:\n  aios signal register {phone} "
                f"--captcha='signalcaptcha://<TOKEN>'"
            )
            raise typer.Exit(code=1)
        print_note(
            f"registration initiated for {phone}; "
            f"check {'phone for voice call' if voice else 'SMS'} for the code"
        )
        render_single(payload)

    run_or_die(_run)


@app.command("verify")
def verify(
    ctx: typer.Context,
    phone: Annotated[str, typer.Argument(help="E.164 phone, e.g. +15551234567.")],
    code: Annotated[str, typer.Argument(help="6-digit verification code from SMS / voice.")],
    pin: Annotated[
        str | None,
        typer.Option("--pin", help="Registration-lock PIN if the account requires one."),
    ] = None,
) -> None:
    """Submit the verification code received by SMS / voice.

    On success, signal-cli writes the account to its on-disk
    ``accounts.json`` and the running connector picks it up without a
    restart on the next ``verify_phone`` call.
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = SignalVerifyRequest(account=phone, code=code, pin=pin)
            result = unwrap(post_connector_signal_verify.sync_detailed(client=client, body=body))
        payload = result.to_dict()
        print_note(f"verified {phone} (uuid={payload.get('uuid', '?')})")
        render_single(payload)

    run_or_die(_run)


@app.command("profile")
def profile(
    ctx: typer.Context,
    phone: Annotated[str, typer.Argument(help="E.164 phone, e.g. +15551234567.")],
    given_name: Annotated[str | None, typer.Option("--given-name")] = None,
    family_name: Annotated[str | None, typer.Option("--family-name")] = None,
    about: Annotated[str | None, typer.Option("--about")] = None,
) -> None:
    """Update profile metadata (given name, family name, about) for a phone.

    Avatar bytes are not exposed in v1 — no operator-side file-staging
    surface yet.
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = SignalProfileRequest(
                account=phone,
                given_name=given_name,
                family_name=family_name,
                about=about,
            )
            result = unwrap(post_connector_signal_profile.sync_detailed(client=client, body=body))
        print_note(f"profile updated for {phone}")
        render_single(result.to_dict())

    run_or_die(_run)
