"""Signal-cli ``register`` / ``verify`` / ``updateProfile`` via the management-call plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aios_connector_http import ManagementHandlerError, management_handler

from .errors import RpcError

if TYPE_CHECKING:
    from .daemon import SignalDaemon

_SIGNAL_CAPTCHA_URL: str = "https://signalcaptchas.org/registration/generate"


def _is_captcha_required(exc: RpcError) -> bool:
    # signal-cli's error shape varies by version; accept the documented code
    # AND a message-substring fallback so the predicate survives upgrades.
    if exc.code == -3:
        return True
    if isinstance(exc.data, dict) and "captcha" in str(exc.data).lower():
        return True
    return "captcha" in str(exc).lower()


class SignalManagementMixin:
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
        try:
            await self._require_daemon().register(phone=account, captcha=captcha, voice=voice)
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
        await self._require_daemon().update_profile(
            phone=account,
            given_name=given_name,
            family_name=family_name,
            about=about,
        )
        return {"account": account}
