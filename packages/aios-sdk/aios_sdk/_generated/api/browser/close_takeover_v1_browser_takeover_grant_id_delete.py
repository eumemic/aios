from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.takeover_close_request import TakeoverCloseRequest
from ...models.takeover_close_response import TakeoverCloseResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    grant_id: str,
    *,
    body: TakeoverCloseRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/browser/takeover/{grant_id}".format(
            grant_id=quote(str(grant_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TakeoverCloseResponse | None:
    if response.status_code == 200:
        response_200 = TakeoverCloseResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | TakeoverCloseResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: TakeoverCloseRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TakeoverCloseResponse]:
    """Close Takeover

     Close a takeover; return the handback (post-human snapshot + inlined
    screenshot + signed-in delta). A browser-dead close still closes → null
    handback fields.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (TakeoverCloseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TakeoverCloseResponse]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: TakeoverCloseRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TakeoverCloseResponse | None:
    """Close Takeover

     Close a takeover; return the handback (post-human snapshot + inlined
    screenshot + signed-in delta). A browser-dead close still closes → null
    handback fields.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (TakeoverCloseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TakeoverCloseResponse
    """

    return sync_detailed(
        grant_id=grant_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: TakeoverCloseRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TakeoverCloseResponse]:
    """Close Takeover

     Close a takeover; return the handback (post-human snapshot + inlined
    screenshot + signed-in delta). A browser-dead close still closes → null
    handback fields.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (TakeoverCloseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TakeoverCloseResponse]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: TakeoverCloseRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TakeoverCloseResponse | None:
    """Close Takeover

     Close a takeover; return the handback (post-human snapshot + inlined
    screenshot + signed-in delta). A browser-dead close still closes → null
    handback fields.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (TakeoverCloseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TakeoverCloseResponse
    """

    return (
        await asyncio_detailed(
            grant_id=grant_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
