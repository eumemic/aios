from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_vault_credential import ListResponseVaultCredential
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    *,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_cursor: None | str | Unset
    if isinstance(cursor, Unset):
        json_cursor = UNSET
    else:
        json_cursor = cursor
    params["cursor"] = json_cursor

    json_limit: int | None | Unset
    if isinstance(limit, Unset):
        json_limit = UNSET
    else:
        json_limit = limit
    params["limit"] = json_limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/vaults/{vault_id}/credentials".format(
            vault_id=quote(str(vault_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseVaultCredential | None:
    if response.status_code == 200:
        response_200 = ListResponseVaultCredential.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseVaultCredential]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseVaultCredential]:
    """List Credentials

     List credentials in a vault, newest first, excluding archived.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Secret
    material is never returned — only metadata (display name, target_url,
    auth_type, timestamps).

    Args:
        vault_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseVaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        cursor=cursor,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseVaultCredential | None:
    """List Credentials

     List credentials in a vault, newest first, excluding archived.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Secret
    material is never returned — only metadata (display name, target_url,
    auth_type, timestamps).

    Args:
        vault_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseVaultCredential
    """

    return sync_detailed(
        vault_id=vault_id,
        client=client,
        cursor=cursor,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseVaultCredential]:
    """List Credentials

     List credentials in a vault, newest first, excluding archived.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Secret
    material is never returned — only metadata (display name, target_url,
    auth_type, timestamps).

    Args:
        vault_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseVaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        cursor=cursor,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseVaultCredential | None:
    """List Credentials

     List credentials in a vault, newest first, excluding archived.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Secret
    material is never returned — only metadata (display name, target_url,
    auth_type, timestamps).

    Args:
        vault_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseVaultCredential
    """

    return (
        await asyncio_detailed(
            vault_id=vault_id,
            client=client,
            cursor=cursor,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
