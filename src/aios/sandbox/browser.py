"""Driver client: invoke ``browser-cli`` inside an account's browser container.

The single seam every worker-side consumer of the browser shares — the
``browser_*`` tool handlers now, the takeover control-plane executor in the
next stacked change. It provisions the account's container on demand and
execs one :class:`~aios.sandbox.browser_protocol.BrowserRequest`, returning
the parsed :class:`~aios.sandbox.browser_protocol.BrowserResponse`.

Failure currency: everything transport-shaped — daemon absent (exit 127 on a
pre-driver deployment), nonzero exit, unparseable stdout, provision failure or
timeout — raises :class:`BrowserUnavailableError`. Callers map it to their own
currency (the tool handlers to ``ToolBail``, the control plane to a failed
call row). An ``ok: false`` response is NOT an error here: the driver spoke,
and the action-level failure is the caller's to render (exit-code contract in
:mod:`aios.sandbox.browser_protocol`).
"""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING

from aios.config import get_settings
from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.browser_protocol import BrowserRequest, BrowserResponse
from aios.sandbox.spec import BrowserImageUnconfiguredError

if TYPE_CHECKING:
    from aios.sandbox.registry import SandboxRegistry


class BrowserUnavailableError(Exception):
    """The account's browser container or driver could not be reached.

    Transport-level only — never an action failure. Deliberately NOT a bare
    passthrough of :class:`~aios.sandbox.backends.base.SandboxBackendError`:
    tool dispatch evicts the CALLING session's sandbox on unrecognized
    exceptions, and a browser-container fault must never be treated as the
    caller's sandbox being unhealthy.
    """


async def driver_call(
    registry: SandboxRegistry,
    account_id: str,
    request: BrowserRequest,
    *,
    timeout_s: int,
) -> BrowserResponse:
    """Provision-if-needed and run one driver request; return the response.

    ``registry`` is the worker's ``SandboxRegistry`` (typed loosely to keep
    this module import-light; it only calls ``get_or_provision_browser`` and
    ``exec``). ``timeout_s`` is the in-container deadline for THIS op —
    :data:`~aios.config.Settings.sandbox_browser_action_timeout_seconds` for
    actions, the longer takeover-open budget for the drain-blocking open.

    Output is bounded by ``sandbox_browser_exec_max_output_bytes`` — never
    ``bash_max_output_bytes``, whose 100 KB default would truncate a
    snapshot-bearing JSON envelope mid-document.
    """
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.sandbox_browser_provision_timeout_seconds):
            handle = await registry.get_or_provision_browser(account_id)
    except TimeoutError as err:
        raise BrowserUnavailableError(
            f"browser container for {account_id} did not provision within "
            f"{settings.sandbox_browser_provision_timeout_seconds:.0f}s"
        ) from err
    except BrowserImageUnconfiguredError:
        # Not a transport fault: the deployment has no browser image at all.
        # Passes through distinctly so callers can render "not enabled here"
        # rather than "try again shortly" (retrying will never help).
        raise
    except SandboxBackendError as err:
        raise BrowserUnavailableError(str(err)) from err

    command = f"browser-cli invoke {shlex.quote(request.model_dump_json())}"
    try:
        result = await registry.exec(
            handle,
            command,
            timeout_seconds=timeout_s,
            max_output_bytes=settings.sandbox_browser_exec_max_output_bytes,
        )
    except SandboxBackendError as err:
        raise BrowserUnavailableError(str(err)) from err

    if result.timed_out:
        raise BrowserUnavailableError(
            f"driver did not respond within {timeout_s}s (op={request.op})"
        )
    if result.exit_code != 0:
        # Exit-code contract: nonzero = the driver never produced a response
        # document (binary absent → 127; daemon down; crash mid-op).
        raise BrowserUnavailableError(
            f"browser-cli exited {result.exit_code} (op={request.op}): "
            f"{result.stderr.strip()[:500]}"
        )
    try:
        return BrowserResponse.model_validate_json(result.stdout)
    except ValueError as err:
        raise BrowserUnavailableError(
            f"driver produced an unparseable response (op={request.op}): {err}"
        ) from err
