from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    resource_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/sessions/{session_id}/resources/{resource_id}".format(
            session_id=quote(str(session_id), safe=""),
            resource_id=quote(str(resource_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

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
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Remove Resource

     Detach a single resource by id. Granular remove-one operation per
    #270. A ``memstore_`` id detaches that memory store (the id IS the
    memory_store_id); a ``ghrepo_`` id detaches that attachment and purges
    its working tree. A malformed/unknown-prefix id is a 4xx.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        resource_id=resource_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Remove Resource

     Detach a single resource by id. Granular remove-one operation per
    #270. A ``memstore_`` id detaches that memory store (the id IS the
    memory_store_id); a ``ghrepo_`` id detaches that attachment and purges
    its working tree. A malformed/unknown-prefix id is a 4xx.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        resource_id=resource_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Remove Resource

     Detach a single resource by id. Granular remove-one operation per
    #270. A ``memstore_`` id detaches that memory store (the id IS the
    memory_store_id); a ``ghrepo_`` id detaches that attachment and purges
    its working tree. A malformed/unknown-prefix id is a 4xx.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        resource_id=resource_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Remove Resource

     Detach a single resource by id. Granular remove-one operation per
    #270. A ``memstore_`` id detaches that memory store (the id IS the
    memory_store_id); a ``ghrepo_`` id detaches that attachment and purges
    its working tree. A malformed/unknown-prefix id is a 4xx.

    Args:
        session_id (str):
        resource_id (str):
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
            resource_id=resource_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
