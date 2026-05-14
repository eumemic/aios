"""Operator-initiated signal management handlers (#348).

Closes the SSH-required gap for signal-cli's ``register`` / ``verify``
/ ``updateProfile`` JSON-RPC methods.  The aios api dispatches operator
calls through the SDK's
:class:`~aios_connector_http.HttpConnector._management_call_loop` SSE,
which routes to the methods on this mixin by their ``@management_handler``
name.  Each handler is a thin layer over the corresponding
:class:`~aios_signal.daemon.SignalDaemon` wrapper, with one bit of
domain-specific logic: signal-cli surfaces captcha-required as an RPC
error carrying a ``captcha`` URL in its ``data`` field, and the
handler unwraps that into a :class:`ManagementHandlerError` payload so
the operator-facing route returns a 200 actionable response instead of
burying the URL in an error envelope.

Kept out of :mod:`aios_signal.connector` so that file stays focused on
inbound dispatch + outbound tool surface.
:class:`~aios_signal.connector.SignalConnector` mixes this class in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aios_connector_http import ManagementHandlerError, management_handler

from .errors import RpcError

if TYPE_CHECKING:
    from .daemon import SignalDaemon

# Canonical signal registration captcha endpoint.  Operators solve in a
# browser, copy the resulting ``signalcaptcha://`` link, and retry
# register with it as the ``captcha`` field.
_SIGNAL_CAPTCHA_URL: str = "https://signalcaptchas.org/registration/generate"


def _is_captcha_required(exc: RpcError) -> bool:
    """True iff a signal-cli ``register`` RpcError is captcha-required.

    Signal-cli's response shape is not formally documented; the
    indicators we've observed are:

    * ``code = -3`` with ``data`` carrying a ``captcha_required`` token
    * a message string mentioning ``captcha`` in some form

    We accept both signals — code-based when present, substring as a
    fallback — to keep the predicate robust across signal-cli versions.
    """
    if exc.code is not None and exc.code in (-3,):
        return True
    if isinstance(exc.data, dict) and "captcha" in str(exc.data).lower():
        return True
    return "captcha" in str(exc).lower()


class SignalManagementMixin:
    """Mixin adding ``@management_handler``-decorated methods to ``SignalConnector``.

    Expects the host class to expose ``self._daemon: SignalDaemon | None``
    (the live daemon facade) and to have called :meth:`setup` on the
    runner first.  Calling a handler before ``setup`` raises
    :class:`RuntimeError`; the operator-facing route surfaces this as
    a 502 via :class:`~aios.errors.ConnectorCallFailedError`.
    """

    _daemon: SignalDaemon | None

    def _require_daemon(self) -> SignalDaemon:
        if self._daemon is None:
            raise RuntimeError(
                "signal daemon not started; setup() must run before management calls"
            )
        return self._daemon

    @management_handler()
    async def register(
        self,
        *,
        account: str,
        captcha: str | None = None,
        voice: bool = False,
    ) -> dict[str, Any]:
        """Initiate signal-cli registration for ``account``.

        Captcha-required surfaces via :class:`ManagementHandlerError`
        carrying ``status="captcha_required"`` plus the signalcaptcha
        URL; the operator-facing route renders that as a 200 response
        with an actionable next step.
        """
        try:
            await self._require_daemon().register(
                phone=account, captcha=captcha, voice=voice
            )
        except RpcError as exc:
            if _is_captcha_required(exc):
                raise ManagementHandlerError(
                    {
                        "status": "captcha_required",
                        "captcha_url": _SIGNAL_CAPTCHA_URL,
                        "account": account,
                    }
                ) from exc
            raise
        return {"account": account, "status": "voice_sent" if voice else "sms_sent"}

    @management_handler()
    async def verify(
        self,
        *,
        account: str,
        code: str,
        pin: str | None = None,
    ) -> dict[str, Any]:
        """Submit the SMS / voice verification code for ``account``.

        signal-cli's ``verify`` result is ``{"uuid": ...}`` on success;
        we surface that verbatim.  Failures (wrong code, expired
        challenge) raise :class:`RpcError` and the SDK dispatcher
        translates them into a generic ``{"error": ...}`` envelope.
        """
        result = await self._require_daemon().verify(phone=account, code=code, pin=pin)
        return {"account": account, "uuid": result.get("uuid", "")}

    @management_handler(method="updateProfile")
    async def update_profile(
        self,
        *,
        account: str,
        given_name: str | None = None,
        family_name: str | None = None,
        about: str | None = None,
    ) -> dict[str, Any]:
        """Update profile metadata for ``account``.

        Avatar bytes are not yet supported (no operator-side file
        staging surface); only text fields flow through.
        """
        await self._require_daemon().update_profile(
            phone=account,
            given_name=given_name,
            family_name=family_name,
            about=about,
        )
        return {"account": account}
