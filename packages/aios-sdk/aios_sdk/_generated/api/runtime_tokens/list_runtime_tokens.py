from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_runtime_token import ListResponseRuntimeToken
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    connector: str,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["connector"] = connector

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/runtime-tokens",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseRuntimeToken | None:
    if response.status_code == 200:
        response_200 = ListResponseRuntimeToken.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseRuntimeToken]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    connector: str,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseRuntimeToken]:
    """List

     All tokens (revoked included) for ``connector``, newest first.

    Args:
        connector (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseRuntimeToken]
    """

    kwargs = _get_kwargs(
        connector=connector,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    connector: str,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseRuntimeToken | None:
    """List

     All tokens (revoked included) for ``connector``, newest first.

    Args:
        connector (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseRuntimeToken
    """

    return sync_detailed(
        client=client,
        connector=connector,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    connector: str,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseRuntimeToken]:
    """List

     All tokens (revoked included) for ``connector``, newest first.

    Args:
        connector (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseRuntimeToken]
    """

    kwargs = _get_kwargs(
        connector=connector,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    connector: str,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseRuntimeToken | None:
    """List

     All tokens (revoked included) for ``connector``, newest first.

    Args:
        connector (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseRuntimeToken
    """

    return (
        await asyncio_detailed(
            client=client,
            connector=connector,
            authorization=authorization,
        )
    ).parsed
