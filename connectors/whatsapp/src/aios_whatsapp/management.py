"""WhatsApp pairing management via the @management_handler plane.

Operator → AIOS server route → management-call queue → this mixin's
methods → per-connection daemon RPC.  Mirrors
``aios_signal.management``'s pattern; the difference is daemon scope —
WhatsApp's daemon is per-connection so we look up the right one by
matching the operator's phone against ``self.state``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from aios_connector_http import ManagementHandlerError, management_handler

from .prompts import _jid_identity_key

if TYPE_CHECKING:
    from .connector import _WhatsappConnectionState

log = structlog.get_logger(__name__)


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
        """
        state = self._state_for_phone(external_account_id)
        outcome = await state.daemon.confirm_pairing()
        # `or "error"` covers both missing key and the empty-string the
        # daemon could emit if outcome was wiped mid-race; the route then
        # maps an unrecognized status to an "error" response with the raw
        # value in reason.
        status = outcome.get("status") or "error"
        # Security gate: a device linked to the wrong WhatsApp account is
        # live exposure — the operator would be sending/receiving as an
        # account they don't control.  Whatsmeow reports the scanned
        # device's jid; if its phone part doesn't match the connection's
        # external_account_id we must NEVER persist the pairing.  We verify
        # EVERY success: a success with no jid is unverifiable, which is just
        # as dangerous as a mismatch, so it takes the same hard-fail +
        # auto-unpair path.
        if status == "success":
            jid = outcome.get("jid") or ""
            # `_jid_identity_key` strips the `@host` and the `:D` device
            # suffix to the phone-bearing local part — the single source of
            # truth for JID parsing in this package.  An empty jid yields "",
            # which can never equal `expected`, so it fails closed.
            scanned = _jid_identity_key(jid)
            expected = normalize_phone(external_account_id).removeprefix("+")
            # No special-case for non-phone JID spaces (e.g. WhatsApp LID —
            # `@lid` identities that live in a separate identifier space and
            # cannot be mapped to a phone number): `_jid_identity_key` would
            # produce a value that can never equal a phone-digit `expected`,
            # so they fall into the fail-closed branch below BY DESIGN.
            # Treating an unverifiable identity as a failure (unpair) rather
            # than passing it through is the whole point of this gate — a
            # fail-open path would reopen the exact wrong-account bypass it
            # closes.  A LID-aware verification path, should the daemon ever
            # report a LID self-JID at pairing, is tracked as a followup.
            if scanned != expected:
                if scanned:
                    mismatch = (
                        f"paired_jid_mismatch: scanned account +{scanned} does not "
                        f"match connection +{expected}"
                    )
                else:
                    mismatch = (
                        "paired_jid_mismatch: daemon reported success without a JID, "
                        f"cannot verify against connection +{expected}"
                    )
                status = "error"
                try:
                    await state.daemon.unpair()
                    reason = f"{mismatch}; device has been unpaired"
                    unpaired = True
                except Exception as exc:
                    reason = (
                        f"{mismatch}; auto-unpair ALSO failed ({exc!r}) "
                        "— device may still be paired"
                    )
                    unpaired = False
                # A wrong-account linkage (or an unverifiable success) is a
                # security event — possible hijack attempt or misconfigured
                # connection.  Surface it server-side; the response only
                # reaches the operator.
                log.warning(
                    "whatsapp_pairing_jid_mismatch",
                    external_account_id=external_account_id,
                    jid=outcome.get("jid"),
                    expected=expected,
                    unpaired=unpaired,
                )
                # Surface the mismatch reason and, when present, the offending
                # jid; drop push_name so a failed pairing can't masquerade as
                # a successful one via a display name.  No jid key when the
                # daemon gave us none — don't fabricate one.
                outcome = {"reason": reason}
                if jid:
                    outcome["jid"] = jid
        response: dict[str, Any] = {
            "external_account_id": external_account_id,
            "status": status,
        }
        for key in ("jid", "push_name", "reason"):
            # Use `is not None` so future fields with a falsy-but-
            # meaningful value (0, False) aren't silently dropped.
            if (value := outcome.get(key)) is not None:
                response[key] = value
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
