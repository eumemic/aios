from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.who_am_i import WhoAmI
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connector-tokens/whoami",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WhoAmI | None:
    if response.status_code == 200:
        response_200 = WhoAmI.from_dict(response.json())

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
) -> Response[HTTPValidationError | WhoAmI]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WhoAmI]:
    """Whoami

     Resolve the bearer token to its ``connection_id``.

    Sanity check / debugging endpoint for connector containers — call
    once at startup to confirm the token is valid and points where the
    operator intended.  Authed by ``ConnectorAuthDep`` (token, NOT
    operator key), so any side-channel access from the operator surface
    is impossible.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WhoAmI]
    """

    kwargs = _get_kwargs(
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WhoAmI | None:
    """Whoami

     Resolve the bearer token to its ``connection_id``.

    Sanity check / debugging endpoint for connector containers — call
    once at startup to confirm the token is valid and points where the
    operator intended.  Authed by ``ConnectorAuthDep`` (token, NOT
    operator key), so any side-channel access from the operator surface
    is impossible.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WhoAmI
    """

    return sync_detailed(
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WhoAmI]:
    """Whoami

     Resolve the bearer token to its ``connection_id``.

    Sanity check / debugging endpoint for connector containers — call
    once at startup to confirm the token is valid and points where the
    operator intended.  Authed by ``ConnectorAuthDep`` (token, NOT
    operator key), so any side-channel access from the operator surface
    is impossible.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WhoAmI]
    """

    kwargs = _get_kwargs(
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WhoAmI | None:
    """Whoami

     Resolve the bearer token to its ``connection_id``.

    Sanity check / debugging endpoint for connector containers — call
    once at startup to confirm the token is valid and points where the
    operator intended.  Authed by ``ConnectorAuthDep`` (token, NOT
    operator key), so any side-channel access from the operator surface
    is impossible.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WhoAmI
    """

    return (
        await asyncio_detailed(
            client=client,
            authorization=authorization,
        )
    ).parsed
