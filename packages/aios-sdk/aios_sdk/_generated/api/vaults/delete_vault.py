from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.vault import Vault
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/vaults/{vault_id}".format(
            vault_id=quote(str(vault_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Vault | None:
    if response.status_code == 200:
        response_200 = Vault.from_dict(response.json())

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
) -> Response[HTTPValidationError | Vault]:
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
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Vault]:
    """Delete

     Soft-archive a vault (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the vault from default lists, zeroing the
    encrypted secret material of its credentials (same behavior as
    ``archive_vault``). The rows persist for audit and history is retained.
    Bare DELETE is never silently destructive; for the irreversible
    hard-delete use ``POST /v1/vaults/{vault_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Vault]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
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
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Vault | None:
    """Delete

     Soft-archive a vault (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the vault from default lists, zeroing the
    encrypted secret material of its credentials (same behavior as
    ``archive_vault``). The rows persist for audit and history is retained.
    Bare DELETE is never silently destructive; for the irreversible
    hard-delete use ``POST /v1/vaults/{vault_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Vault
    """

    return sync_detailed(
        vault_id=vault_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Vault]:
    """Delete

     Soft-archive a vault (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the vault from default lists, zeroing the
    encrypted secret material of its credentials (same behavior as
    ``archive_vault``). The rows persist for audit and history is retained.
    Bare DELETE is never silently destructive; for the irreversible
    hard-delete use ``POST /v1/vaults/{vault_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Vault]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Vault | None:
    """Delete

     Soft-archive a vault (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the vault from default lists, zeroing the
    encrypted secret material of its credentials (same behavior as
    ``archive_vault``). The rows persist for audit and history is retained.
    Bare DELETE is never silently destructive; for the irreversible
    hard-delete use ``POST /v1/vaults/{vault_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Vault
    """

    return (
        await asyncio_detailed(
            vault_id=vault_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
