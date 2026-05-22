from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    after_seq: int | Unset = 0,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["after_seq"] = after_seq

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/stream".format(
            session_id=quote(str(session_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = response.json()
        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Stream Events

     Stream session events as Server-Sent Events.

    Preflights the LISTEN connection BEFORE constructing the response
    (issue #376): a transient ``asyncpg.connect`` failure during
    testcontainer warmup or a brief Postgres outage surfaces as a clean
    503 with proper headers rather than a half-open chunked stream
    after 200 OK has gone out.

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        after_seq=after_seq,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Stream Events

     Stream session events as Server-Sent Events.

    Preflights the LISTEN connection BEFORE constructing the response
    (issue #376): a transient ``asyncpg.connect`` failure during
    testcontainer warmup or a brief Postgres outage surfaces as a clean
    503 with proper headers rather than a half-open chunked stream
    after 200 OK has gone out.

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        after_seq=after_seq,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Stream Events

     Stream session events as Server-Sent Events.

    Preflights the LISTEN connection BEFORE constructing the response
    (issue #376): a transient ``asyncpg.connect`` failure during
    testcontainer warmup or a brief Postgres outage surfaces as a clean
    503 with proper headers rather than a half-open chunked stream
    after 200 OK has gone out.

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        after_seq=after_seq,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Stream Events

     Stream session events as Server-Sent Events.

    Preflights the LISTEN connection BEFORE constructing the response
    (issue #376): a transient ``asyncpg.connect`` failure during
    testcontainer warmup or a brief Postgres outage surfaces as a clean
    503 with proper headers rather than a half-open chunked stream
    after 200 OK has gone out.

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            after_seq=after_seq,
            authorization=authorization,
        )
    ).parsed
