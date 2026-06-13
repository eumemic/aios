"""``aios whatsapp {start-pairing, pairing-code, confirm-pairing, unpair}`` — operator pairing ops."""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands._shared import render_single, unwrap
from aios.cli.coverage import covers
from aios.cli.output import print_note
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.connectors import (
    post_connector_whatsapp_confirm_pairing,
    post_connector_whatsapp_pairing_code,
    post_connector_whatsapp_start_pairing,
    post_connector_whatsapp_unpair,
)
from aios_sdk._generated.models.whatsapp_confirm_pairing_request import (
    WhatsappConfirmPairingRequest,
)
from aios_sdk._generated.models.whatsapp_pairing_code_request import (
    WhatsappPairingCodeRequest,
)
from aios_sdk._generated.models.whatsapp_start_pairing_request import (
    WhatsappStartPairingRequest,
)
from aios_sdk._generated.models.whatsapp_unpair_request import WhatsappUnpairRequest

app = typer.Typer(
    name="whatsapp",
    help="WhatsApp management operations (pair, pairing-code, confirm-pairing, unpair).",
    no_args_is_help=True,
)


@app.command("start-pairing")
@covers("post_connector_whatsapp_start_pairing")
def start_pairing(
    ctx: typer.Context,
    phone: Annotated[
        str,
        typer.Argument(help="E.164 phone of the WhatsApp connection, e.g. +15551234567."),
    ],
) -> None:
    """Begin a QR pairing session for the given phone's WhatsApp connection.

    Returns the pairing code (a ``wa.me/settings/linked_devices#<...>``
    URL) which the operator scans from WhatsApp → Settings → Linked
    Devices → Link a Device.  After scanning, run ``aios whatsapp
    confirm-pairing <phone>`` to block until WhatsApp's servers
    acknowledge the link.
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = WhatsappStartPairingRequest(external_account_id=phone)
            result = unwrap(
                post_connector_whatsapp_start_pairing.sync_detailed(client=client, body=body)
            )
        # Defensive: if the API ever switches to 204 No Content,
        # ``unwrap`` returns None and ``result.to_dict()`` would
        # AttributeError.  Surface a clean note instead.
        payload = result.to_dict() if result is not None else {}
        print_note(
            f"pairing started for {phone}; scan the QR code (WhatsApp → "
            f"Settings → Linked Devices → Link a Device), then run "
            f"`aios whatsapp confirm-pairing {phone}`"
        )
        if payload:
            render_single(payload)

    run_or_die(_run)


@app.command("pairing-code")
@covers("post_connector_whatsapp_pairing_code")
def pairing_code(
    ctx: typer.Context,
    phone: Annotated[str, typer.Argument(help="E.164 phone of the WhatsApp connection.")],
) -> None:
    """Fetch the QR code currently live for the in-flight pairing attempt.

    whatsmeow rotates the code ~every 20 s over the ~100 s attempt;
    ``start-pairing`` returns only the first.  Poll this every few
    seconds and re-render the QR whenever ``rotation_seq`` changes so the
    operator always has a scannable code, not just the first ~20 s window.
    Fails if no attempt is live (none started, or already terminated).
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = WhatsappPairingCodeRequest(external_account_id=phone)
            result = unwrap(
                post_connector_whatsapp_pairing_code.sync_detailed(client=client, body=body)
            )
        payload = result.to_dict() if result is not None else {}
        render_single(payload)

    run_or_die(_run)


@app.command("confirm-pairing")
@covers("post_connector_whatsapp_confirm_pairing")
def confirm_pairing(
    ctx: typer.Context,
    phone: Annotated[str, typer.Argument(help="E.164 phone of the WhatsApp connection.")],
) -> None:
    """Block until the pairing flow terminates.

    Returns ``status=success`` once WhatsApp acknowledges the device
    link, or ``status=timeout``/``status=error`` if the QR window
    expires or the daemon's whatsmeow client rejects the handshake.
    The connector picks up the new device identity immediately; no
    restart needed.
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = WhatsappConfirmPairingRequest(external_account_id=phone)
            result = unwrap(
                post_connector_whatsapp_confirm_pairing.sync_detailed(client=client, body=body)
            )
        payload = result.to_dict() if result is not None else {}
        status = payload.get("status", "?")
        if status == "success":
            print_note(f"paired {phone} (jid={payload.get('jid', '?')})")
        else:
            print_note(f"pairing for {phone} ended with status={status}")
        render_single(payload)

    run_or_die(_run)


@app.command("unpair")
@covers("post_connector_whatsapp_unpair")
def unpair(
    ctx: typer.Context,
    phone: Annotated[str, typer.Argument(help="E.164 phone of the WhatsApp connection.")],
) -> None:
    """Unlink the WhatsApp device server-side and clear the local store.

    Idempotent at the HTTP layer (204 No Content on success).  The
    connector's daemon swaps in a fresh whatsmeow.Client behind an
    atomic.Pointer so the next ``start-pairing`` works in the same
    daemon process without a restart.
    """

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            body = WhatsappUnpairRequest(external_account_id=phone)
            unwrap(post_connector_whatsapp_unpair.sync_detailed(client=client, body=body))
        print_note(f"unpaired {phone}")

    run_or_die(_run)
