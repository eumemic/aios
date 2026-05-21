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
        """
        state = self._state_for_phone(external_account_id)
        result = await state.daemon.start_pairing()
        return {"external_account_id": external_account_id, "code": result.get("code", "")}

    @management_handler()
    async def confirmPairing(self, *, external_account_id: str) -> dict[str, Any]:
        """Block until pairing terminates.

        Returns ``{external_account_id, status, jid?, push_name?, reason?}``.
        ``status`` is ``"success"``, ``"timeout"``, or ``"error"``.
        """
        state = self._state_for_phone(external_account_id)
        outcome = await state.daemon.confirm_pairing()
        response: dict[str, Any] = {
            "external_account_id": external_account_id,
            "status": outcome.get("status", ""),
        }
        for key in ("jid", "push_name", "reason"):
            if value := outcome.get(key):
                response[key] = value
        return response

    @management_handler()
    async def unpair(self, *, external_account_id: str) -> dict[str, Any]:
        """Log out the WhatsApp device on the server."""
        state = self._state_for_phone(external_account_id)
        await state.daemon.unpair()
        return {"external_account_id": external_account_id, "status": "ok"}

    def _state_for_phone(self, phone: str) -> _WhatsappConnectionState:
        for state in self.state.values():
            if state.phone == phone:
                return state
        raise ManagementHandlerError(
            {
                "status": "no_active_connection",
                "external_account_id": phone,
            }
        )
