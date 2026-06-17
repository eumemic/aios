from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.ingest_external_event_response_ingest_external_event import (
    IngestExternalEventResponseIngestExternalEvent,
)
from ...types import Response


def _get_kwargs(
    ingest_token: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/triggers/ingest/{ingest_token}".format(
            ingest_token=quote(str(ingest_token), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | IngestExternalEventResponseIngestExternalEvent | None:
    if response.status_code == 202:
        response_202 = IngestExternalEventResponseIngestExternalEvent.from_dict(
            response.json()
        )

        return response_202

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | IngestExternalEventResponseIngestExternalEvent]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    ingest_token: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[HTTPValidationError | IngestExternalEventResponseIngestExternalEvent]:
    r"""Ingest External Event

     Resolve a per-trigger ingest token, record a pending fire, and dispatch.

    ``PoolDep`` only — deliberately NO ``AccountIdDep``: this is the sole
    account-key-free route; the path token authenticates and scopes the call.

    Returns ``202 {\"trigger_run_id\": …}``. A lost post-commit defer is recovered
    by the existing ``list_pending_trigger_run_refs`` sweep — identical
    resilience to run_completion, for free.

    Args:
        ingest_token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | IngestExternalEventResponseIngestExternalEvent]
    """

    kwargs = _get_kwargs(
        ingest_token=ingest_token,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    ingest_token: str,
    *,
    client: AuthenticatedClient | Client,
) -> HTTPValidationError | IngestExternalEventResponseIngestExternalEvent | None:
    r"""Ingest External Event

     Resolve a per-trigger ingest token, record a pending fire, and dispatch.

    ``PoolDep`` only — deliberately NO ``AccountIdDep``: this is the sole
    account-key-free route; the path token authenticates and scopes the call.

    Returns ``202 {\"trigger_run_id\": …}``. A lost post-commit defer is recovered
    by the existing ``list_pending_trigger_run_refs`` sweep — identical
    resilience to run_completion, for free.

    Args:
        ingest_token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | IngestExternalEventResponseIngestExternalEvent
    """

    return sync_detailed(
        ingest_token=ingest_token,
        client=client,
    ).parsed


async def asyncio_detailed(
    ingest_token: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[HTTPValidationError | IngestExternalEventResponseIngestExternalEvent]:
    r"""Ingest External Event

     Resolve a per-trigger ingest token, record a pending fire, and dispatch.

    ``PoolDep`` only — deliberately NO ``AccountIdDep``: this is the sole
    account-key-free route; the path token authenticates and scopes the call.

    Returns ``202 {\"trigger_run_id\": …}``. A lost post-commit defer is recovered
    by the existing ``list_pending_trigger_run_refs`` sweep — identical
    resilience to run_completion, for free.

    Args:
        ingest_token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | IngestExternalEventResponseIngestExternalEvent]
    """

    kwargs = _get_kwargs(
        ingest_token=ingest_token,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    ingest_token: str,
    *,
    client: AuthenticatedClient | Client,
) -> HTTPValidationError | IngestExternalEventResponseIngestExternalEvent | None:
    r"""Ingest External Event

     Resolve a per-trigger ingest token, record a pending fire, and dispatch.

    ``PoolDep`` only — deliberately NO ``AccountIdDep``: this is the sole
    account-key-free route; the path token authenticates and scopes the call.

    Returns ``202 {\"trigger_run_id\": …}``. A lost post-commit defer is recovered
    by the existing ``list_pending_trigger_run_refs`` sweep — identical
    resilience to run_completion, for free.

    Args:
        ingest_token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | IngestExternalEventResponseIngestExternalEvent
    """

    return (
        await asyncio_detailed(
            ingest_token=ingest_token,
            client=client,
        )
    ).parsed
