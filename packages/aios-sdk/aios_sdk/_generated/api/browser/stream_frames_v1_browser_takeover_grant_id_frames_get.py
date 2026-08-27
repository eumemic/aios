from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    grant_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/browser/takeover/{grant_id}/frames".format(
            grant_id=quote(str(grant_id), safe=""),
        ),
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
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Stream Frames

     Stream the takeover screencast as SSE ``frame`` events, ending on close.

    Novel among aios SSE routes: it tails a shared-filesystem ring, not a
    LISTEN channel (the driver has no route to Postgres). The frames dir is
    derived server-side from the scoped grant's account — the client supplies
    only ``grant_id`` — so path containment holds by construction.

    Args:
        grant_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
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
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Stream Frames

     Stream the takeover screencast as SSE ``frame`` events, ending on close.

    Novel among aios SSE routes: it tails a shared-filesystem ring, not a
    LISTEN channel (the driver has no route to Postgres). The frames dir is
    derived server-side from the scoped grant's account — the client supplies
    only ``grant_id`` — so path containment holds by construction.

    Args:
        grant_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        grant_id=grant_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Stream Frames

     Stream the takeover screencast as SSE ``frame`` events, ending on close.

    Novel among aios SSE routes: it tails a shared-filesystem ring, not a
    LISTEN channel (the driver has no route to Postgres). The frames dir is
    derived server-side from the scoped grant's account — the client supplies
    only ``grant_id`` — so path containment holds by construction.

    Args:
        grant_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Stream Frames

     Stream the takeover screencast as SSE ``frame`` events, ending on close.

    Novel among aios SSE routes: it tails a shared-filesystem ring, not a
    LISTEN channel (the driver has no route to Postgres). The frames dir is
    derived server-side from the scoped grant's account — the client supplies
    only ``grant_id`` — so path containment holds by construction.

    Args:
        grant_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            grant_id=grant_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
