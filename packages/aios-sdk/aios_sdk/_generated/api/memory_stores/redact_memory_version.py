from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.memory_version import MemoryVersion
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    version_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/memory-stores/{store_id}/memory-versions/{version_id}/redact".format(
            store_id=quote(str(store_id), safe=""),
            version_id=quote(str(version_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | MemoryVersion | None:
    if response.status_code == 200:
        response_200 = MemoryVersion.from_dict(response.json())

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
) -> Response[HTTPValidationError | MemoryVersion]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    store_id: str,
    version_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MemoryVersion]:
    """Redact Version

     Redact the content of a historical memory version in place.

    The version row persists for audit (with the actor and timestamp) but
    its content field is cleared. Use to scrub sensitive data that was
    previously written into a memory. Live memory content is unaffected
    (only this specific historical version is redacted).

    Args:
        store_id (str):
        version_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MemoryVersion]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        version_id=version_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    store_id: str,
    version_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MemoryVersion | None:
    """Redact Version

     Redact the content of a historical memory version in place.

    The version row persists for audit (with the actor and timestamp) but
    its content field is cleared. Use to scrub sensitive data that was
    previously written into a memory. Live memory content is unaffected
    (only this specific historical version is redacted).

    Args:
        store_id (str):
        version_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MemoryVersion
    """

    return sync_detailed(
        store_id=store_id,
        version_id=version_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    store_id: str,
    version_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MemoryVersion]:
    """Redact Version

     Redact the content of a historical memory version in place.

    The version row persists for audit (with the actor and timestamp) but
    its content field is cleared. Use to scrub sensitive data that was
    previously written into a memory. Live memory content is unaffected
    (only this specific historical version is redacted).

    Args:
        store_id (str):
        version_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MemoryVersion]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        version_id=version_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    store_id: str,
    version_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MemoryVersion | None:
    """Redact Version

     Redact the content of a historical memory version in place.

    The version row persists for audit (with the actor and timestamp) but
    its content field is cleared. Use to scrub sensitive data that was
    previously written into a memory. Live memory content is unaffected
    (only this specific historical version is redacted).

    Args:
        store_id (str):
        version_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MemoryVersion
    """

    return (
        await asyncio_detailed(
            store_id=store_id,
            version_id=version_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
