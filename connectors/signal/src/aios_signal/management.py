"""Signal-cli ``register`` / ``verify`` / ``updateProfile`` via the management-call plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aios_connector_http import ManagementHandlerError, management_handler

from .errors import RpcError

if TYPE_CHECKING:
    from .daemon import SignalDaemon

_SIGNAL_CAPTCHA_URL: str = "https://signalcaptchas.org/registration/generate"


def _is_captcha_required(exc: RpcError) -> bool:
    return exc.code == -3


def _captcha_url(exc: RpcError) -> str:
    # Prefer the URL signal-cli emits in the structured ``data`` field (its
    # exact key varies by version); fall back to the canonical endpoint.
    if isinstance(exc.data, dict):
        for key in ("captcha_url", "captcha"):
            value = exc.data.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
    return _SIGNAL_CAPTCHA_URL


class SignalManagementMixin:
    _daemon: SignalDaemon | None

    @management_handler()
    async def register(
        self,
        *,
        external_account_id: str,
        captcha: str | None = None,
        voice: bool = False,
    ) -> dict[str, Any]:
        assert self._daemon is not None
        try:
            await self._daemon.register(phone=external_account_id, captcha=captcha, voice=voice)
        except RpcError as exc:
            if _is_captcha_required(exc):
                raise ManagementHandlerError(
                    {
                        "status": "captcha_required",
                        "captcha_url": _captcha_url(exc),
                        "external_account_id": external_account_id,
                    }
                ) from exc
            raise
        return {
            "external_account_id": external_account_id,
            "status": "voice_sent" if voice else "sms_sent",
        }

    @management_handler()
    async def verify(
        self,
        *,
        external_account_id: str,
        code: str,
        pin: str | None = None,
    ) -> dict[str, Any]:
        assert self._daemon is not None
        result = await self._daemon.verify(phone=external_account_id, code=code, pin=pin)
        return {"external_account_id": external_account_id, "uuid": result.get("uuid", "")}

    @management_handler(method="updateProfile")
    async def update_profile(
        self,
        *,
        external_account_id: str,
        given_name: str | None = None,
        family_name: str | None = None,
        about: str | None = None,
    ) -> dict[str, Any]:
        assert self._daemon is not None
        await self._daemon.update_profile(
            phone=external_account_id,
            given_name=given_name,
            family_name=family_name,
            about=about,
        )
        return {"external_account_id": external_account_id}
