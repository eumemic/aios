from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.browser_peek_response import BrowserPeekResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    session_id: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_session_id: None | str | Unset
    if isinstance(session_id, Unset):
        json_session_id = UNSET
    else:
        json_session_id = session_id
    params["session_id"] = json_session_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/browser/peek",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> BrowserPeekResponse | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = BrowserPeekResponse.from_dict(response.json())

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
) -> Response[BrowserPeekResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    session_id: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[BrowserPeekResponse | HTTPValidationError]:
    """Browser Peek

     A read-only look at a page: one JPEG of the viewport plus the trusted
    chrome, from ``session_id``'s page when given, else the last-active one.
    Never provisions and never creates a page; refused (409) while a human
    holds the computer.

    Args:
        session_id (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BrowserPeekResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    session_id: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> BrowserPeekResponse | HTTPValidationError | None:
    """Browser Peek

     A read-only look at a page: one JPEG of the viewport plus the trusted
    chrome, from ``session_id``'s page when given, else the last-active one.
    Never provisions and never creates a page; refused (409) while a human
    holds the computer.

    Args:
        session_id (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BrowserPeekResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        session_id=session_id,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    session_id: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[BrowserPeekResponse | HTTPValidationError]:
    """Browser Peek

     A read-only look at a page: one JPEG of the viewport plus the trusted
    chrome, from ``session_id``'s page when given, else the last-active one.
    Never provisions and never creates a page; refused (409) while a human
    holds the computer.

    Args:
        session_id (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BrowserPeekResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    session_id: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> BrowserPeekResponse | HTTPValidationError | None:
    """Browser Peek

     A read-only look at a page: one JPEG of the viewport plus the trusted
    chrome, from ``session_id``'s page when given, else the last-active one.
    Never provisions and never creates a page; refused (409) while a human
    holds the computer.

    Args:
        session_id (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BrowserPeekResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            session_id=session_id,
            authorization=authorization,
        )
    ).parsed
