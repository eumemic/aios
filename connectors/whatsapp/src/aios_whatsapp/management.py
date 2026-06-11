"""WhatsApp pairing management via the @management_handler plane.

Operator → AIOS server route → management-call queue → this mixin's
methods → per-connection daemon RPC.  Mirrors
``aios_signal.management``'s pattern; the difference is daemon scope —
WhatsApp's daemon is per-connection so we look up the right one by
matching the operator's phone against ``self.state``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aios_connector_http import ManagementHandlerError, management_handler

if TYPE_CHECKING:
    from .connector import _WhatsappConnectionState


def normalize_phone(phone: str) -> str:
    """Strip whitespace + common separators and ensure a leading ``+``.

    Used at the two boundaries where operator-provided phones meet
    internal state: ``serve_connection`` normalizes ``secrets["phone"]``
    on store, and :meth:`WhatsappManagementMixin._state_for_phone`
    normalizes ``external_account_id`` on lookup.  Both sides
    canonical-form the input so trivial formatting differences
    (``+15551112222`` vs ``15551112222`` vs ``+1 555 111-2222``) match.
    """
    s = phone.strip().replace("-", "").replace(" ", "")
    if s and not s.startswith("+"):
        s = "+" + s
    return s


class WhatsappManagementMixin:
    # Narrowed by WhatsappConnector to dict[str, _WhatsappConnectionState];
    # declared here so the mixin's helpers type-check on its own.
    state: dict[str, _WhatsappConnectionState]

    @management_handler()
    async def startPairing(self, *, external_account_id: str) -> dict[str, Any]:
        """Initiate QR pairing.  Returns the first QR code.

        The operator scans it within whatsmeow's QR window (~20 s
        before rotation; ~100 s total).  Subsequent QR refreshes are
        drained by the daemon but not surfaced — single-scan pairing
        is the v1 scope.

        Raises :class:`ManagementHandlerError` with a structured
        payload if the daemon returns an empty code (contract bug);
        the AIOS server route surfaces this as a 502 rather than 200
        OK with a blank QR.
        """
        state = self._state_for_phone(external_account_id)
        result = await state.daemon.start_pairing()
        code = result.get("code")
        if not code:
            raise ManagementHandlerError(
                {
                    "status": "error",
                    "external_account_id": external_account_id,
                    "reason": "daemon returned empty pairing code",
                }
            )
        return {"external_account_id": external_account_id, "code": code}

    @management_handler()
    async def confirmPairing(self, *, external_account_id: str) -> dict[str, Any]:
        """Block until pairing terminates.

        Returns ``{external_account_id, status, jid?, push_name?, reason?}``.
        ``status`` is ``"success"``, ``"timeout"``, or ``"error"``.

        On a successful pair we verify the scanned account's JID user
        part matches the connection's phone.  A mismatch means the
        operator scanned the wrong WhatsApp account onto this
        connection: we immediately unpair the device on the server and
        return an error so no foreign device is left paired.
        """
        state = self._state_for_phone(external_account_id)
        outcome = await state.daemon.confirm_pairing()
        response: dict[str, Any] = {
            "external_account_id": external_account_id,
            # `or "error"` covers both missing key and the empty-string
            # the daemon could emit if outcome was wiped mid-race; the
            # route then maps an unrecognized status to an "error"
            # response with the raw value in reason.
            "status": outcome.get("status") or "error",
        }
        for key in ("jid", "push_name", "reason"):
            # Use `is not None` so future fields with a falsy-but-
            # meaningful value (0, False) aren't silently dropped.
            if (value := outcome.get(key)) is not None:
                response[key] = value

        jid = outcome.get("jid")
        if response["status"] == "success" and jid:
            scanned = jid.split("@")[0].split(":")[0]
            # Compare digits-only (mirrors connector._phone_to_jid):
            # normalize_phone strips only spaces/dashes, so a phone
            # stored as "+1 (555) 111-2222" would otherwise never equal
            # the scanned JID's pure-digit user part and a correctly
            # paired device would be falsely auto-unpaired.
            expected = "".join(c for c in external_account_id if c.isdigit())
            if scanned != expected:
                # Wrong account scanned onto this connection.  Unpair
                # the foreign device before returning so we never leave
                # it paired; if the unpair itself fails, say so but
                # still report the mismatch as an error.
                response["status"] = "error"
                response["jid"] = jid
                try:
                    await state.daemon.unpair()
                except Exception as exc:
                    response["reason"] = (
                        f"paired_jid_mismatch: scanned account +{scanned} "
                        f"does not match connection +{expected}; "
                        f"auto-unpair FAILED ({exc}) — device may still be paired"
                    )
                else:
                    response["reason"] = (
                        f"paired_jid_mismatch: scanned account +{scanned} "
                        f"does not match connection +{expected}; "
                        f"device has been unpaired"
                    )
        return response

    @management_handler()
    async def unpair(self, *, external_account_id: str) -> dict[str, Any]:
        """Log out the WhatsApp device on the server."""
        state = self._state_for_phone(external_account_id)
        await state.daemon.unpair()
        return {"external_account_id": external_account_id, "status": "ok"}

    def _state_for_phone(self, phone: str) -> _WhatsappConnectionState:
        target = normalize_phone(phone)
        for state in self.state.values():
            if state.phone == target:
                return state
        raise ManagementHandlerError(
            {
                "status": "no_active_connection",
                "external_account_id": phone,
            }
        )
