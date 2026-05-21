"""E2E: operator-route → NOTIFY → SDK SSE dispatch → result POST → wake → response.

Mocks at the ``@management_handler`` boundary so signal-cli isn't required.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from unittest import mock

import pytest

from aios_connector_http import HttpConnector, ManagementHandlerError, management_handler
from tests.conftest import needs_docker
from tests.e2e.conftest import live_aios_server
from tests.helpers.connections import authed_client, issue_runtime_token


class _FakeSignalConnector(HttpConnector):
    """Real SDK, scripted management handlers.

    Each handler records what it was called with and returns a
    pre-canned response so the test can assert both the request path
    (correct dispatch) and the response path (correct wake) end-to-end.
    """

    connector = "signal"

    def __init__(self, *, base_url: str, token: str) -> None:
        super().__init__(base_url=base_url, token=token)
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.captcha_on_register: bool = False

    @management_handler()
    async def register(
        self, *, external_account_id: str, captcha: str | None = None, voice: bool = False
    ) -> dict:
        self.calls.append(
            (
                "register",
                {
                    "external_account_id": external_account_id,
                    "captcha": captcha,
                    "voice": voice,
                },
            )
        )
        if self.captcha_on_register and captcha is None:
            raise ManagementHandlerError(
                {
                    "status": "captcha_required",
                    "captcha_url": "https://signalcaptchas.org/registration/generate",
                    "external_account_id": external_account_id,
                }
            )
        return {
            "external_account_id": external_account_id,
            "status": "voice_sent" if voice else "sms_sent",
        }

    @management_handler()
    async def verify(self, *, external_account_id: str, code: str, pin: str | None = None) -> dict:
        self.calls.append(
            (
                "verify",
                {"external_account_id": external_account_id, "code": code, "pin": pin},
            )
        )
        return {"external_account_id": external_account_id, "uuid": "u-from-fake"}

    @management_handler(method="updateProfile")
    async def update_profile(
        self,
        *,
        external_account_id: str,
        given_name: str | None = None,
        family_name: str | None = None,
        about: str | None = None,
    ) -> dict:
        self.calls.append(
            (
                "updateProfile",
                {
                    "external_account_id": external_account_id,
                    "given_name": given_name,
                    "family_name": family_name,
                    "about": about,
                },
            )
        )
        return {"external_account_id": external_account_id}


@pytest.fixture
async def live_server(aios_env: dict[str, str]) -> AsyncIterator[str]:
    """Same uvicorn-in-process fixture pattern the other e2e tests use."""
    async with live_aios_server(pool_max_size=8) as url:
        yield url


async def _run_connector(connector: _FakeSignalConnector) -> None:
    """Run the connector with all setup hooks suppressed.

    We don't want the connector's discovery loop (there are no signal
    connections in this test) or tool loop to do anything — the
    management call loop is the only loop we care about.  Skipping
    ``setup()`` keeps the test from needing a daemon facade.
    """
    with mock.patch.object(connector, "_publish_tools_schema", new_callable=mock.AsyncMock):
        await connector.run()


@needs_docker
class TestSignalRegistration:
    async def test_register_then_verify_round_trip(
        self,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Happy path: register returns sms_sent, verify returns the uuid."""
        api_key = aios_env["AIOS_API_KEY"]
        runtime_token = await issue_runtime_token(api_key, live_server, "signal")

        connector = _FakeSignalConnector(base_url=live_server, token=runtime_token)
        connector_task = asyncio.create_task(_run_connector(connector))

        try:
            # Wait for the connector's management SSE to be live.
            await connector.wait_ready()

            async with authed_client(live_server, api_key) as c:
                # Register.
                r = await c.post(
                    "/v1/connectors/signal/register",
                    json={"external_account_id": "+15551234567"},
                    timeout=10.0,
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert body == {
                    "external_account_id": "+15551234567",
                    "status": "sms_sent",
                    "captcha_url": None,
                }

                # Verify.
                r = await c.post(
                    "/v1/connectors/signal/verify",
                    json={"external_account_id": "+15551234567", "code": "123456"},
                    timeout=10.0,
                )
                assert r.status_code == 200, r.text
                assert r.json() == {
                    "external_account_id": "+15551234567",
                    "uuid": "u-from-fake",
                }

                # Profile (only given_name).
                r = await c.post(
                    "/v1/connectors/signal/profile",
                    json={"external_account_id": "+15551234567", "given_name": "Alice"},
                    timeout=10.0,
                )
                assert r.status_code == 204, r.text

            # The handler was hit with the right kwargs.
            assert (
                "register",
                {"external_account_id": "+15551234567", "captcha": None, "voice": False},
            ) in connector.calls
            assert (
                "verify",
                {"external_account_id": "+15551234567", "code": "123456", "pin": None},
            ) in connector.calls
            profile_call = next(c for c in connector.calls if c[0] == "updateProfile")
            assert profile_call[1]["given_name"] == "Alice"
        finally:
            connector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connector_task

    async def test_captcha_required_returns_200_with_url(
        self,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Captcha-required is an actionable state, not an error.

        The connector raises :class:`ManagementHandlerError`; the
        operator-facing route translates the structured payload into a
        200 response carrying ``status="captcha_required"`` and the URL.
        Re-running with the captcha token clears it.
        """
        api_key = aios_env["AIOS_API_KEY"]
        runtime_token = await issue_runtime_token(api_key, live_server, "signal")

        connector = _FakeSignalConnector(base_url=live_server, token=runtime_token)
        connector.captcha_on_register = True
        connector_task = asyncio.create_task(_run_connector(connector))

        try:
            await connector.wait_ready()

            async with authed_client(live_server, api_key) as c:
                r = await c.post(
                    "/v1/connectors/signal/register",
                    json={"external_account_id": "+15559999999"},
                    timeout=10.0,
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["status"] == "captcha_required"
                assert body["captcha_url"] == ("https://signalcaptchas.org/registration/generate")

                # Retry with a captcha token — fake connector accepts any
                # non-None value as "operator solved the captcha."
                r = await c.post(
                    "/v1/connectors/signal/register",
                    json={
                        "external_account_id": "+15559999999",
                        "captcha": "signalcaptcha://solved",
                    },
                    timeout=10.0,
                )
                assert r.status_code == 200, r.text
                assert r.json()["status"] == "sms_sent"
        finally:
            connector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connector_task
