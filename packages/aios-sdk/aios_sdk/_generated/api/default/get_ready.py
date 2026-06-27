from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/ready",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | None:
    if response.status_code == 200:
        return None

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any]:
    r"""Ready

     Readiness probe. Unauthenticated; ``SELECT 1`` under a short timeout.

    Returns 200 ``{\"status\": \"ready\"}`` when Postgres answers AND the
    fail-closed boot-admission gate (#1575) has admitted this process; 503
    ``{\"status\": \"unavailable\"}`` when the pool can't be acquired, the query
    raises, it exceeds the 2 s budget, OR the boot-gate has not yet admitted.
    This is the signal the Docker/compose healthcheck watches, so a silent
    post-startup DB outage becomes a visibly unhealthy container instead of a
    200-returning black hole.

    The boot-gate flag (``app.state.retirements_ok``) stays ``False`` from
    process start until the gate proves the live DB is safe for this code
    (alembic at/past every ``contract_rev`` AND zero live residue). Under a
    rolling deploy this keeps the NEW container unready — so the OLD healthy
    container keeps serving — until the post-deploy migrate lands. Checked
    BEFORE the DB round-trip: an un-admitted process is unready regardless of
    DB liveness.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any]:
    r"""Ready

     Readiness probe. Unauthenticated; ``SELECT 1`` under a short timeout.

    Returns 200 ``{\"status\": \"ready\"}`` when Postgres answers AND the
    fail-closed boot-admission gate (#1575) has admitted this process; 503
    ``{\"status\": \"unavailable\"}`` when the pool can't be acquired, the query
    raises, it exceeds the 2 s budget, OR the boot-gate has not yet admitted.
    This is the signal the Docker/compose healthcheck watches, so a silent
    post-startup DB outage becomes a visibly unhealthy container instead of a
    200-returning black hole.

    The boot-gate flag (``app.state.retirements_ok``) stays ``False`` from
    process start until the gate proves the live DB is safe for this code
    (alembic at/past every ``contract_rev`` AND zero live residue). Under a
    rolling deploy this keeps the NEW container unready — so the OLD healthy
    container keeps serving — until the post-deploy migrate lands. Checked
    BEFORE the DB round-trip: an un-admitted process is unready regardless of
    DB liveness.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)
