from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.vault_credential import VaultCredential
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    credential_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/vaults/{vault_id}/credentials/{credential_id}".format(
            vault_id=quote(str(vault_id), safe=""),
            credential_id=quote(str(credential_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | VaultCredential | None:
    if response.status_code == 200:
        response_200 = VaultCredential.from_dict(response.json())

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
) -> Response[HTTPValidationError | VaultCredential]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    vault_id: str,
    credential_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Delete Credential

     Soft-archive a credential (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the credential from default lists, and zeroes
    its encrypted secret payload (same behavior as
    ``archive_vault_credential``). The row persists for audit. Bare DELETE is
    never silently destructive; for the irreversible hard-delete use
    ``POST /v1/vaults/{vault_id}/credentials/{credential_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        credential_id=credential_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    vault_id: str,
    credential_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Delete Credential

     Soft-archive a credential (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the credential from default lists, and zeroes
    its encrypted secret payload (same behavior as
    ``archive_vault_credential``). The row persists for audit. Bare DELETE is
    never silently destructive; for the irreversible hard-delete use
    ``POST /v1/vaults/{vault_id}/credentials/{credential_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | VaultCredential
    """

    return sync_detailed(
        vault_id=vault_id,
        credential_id=credential_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    vault_id: str,
    credential_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Delete Credential

     Soft-archive a credential (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the credential from default lists, and zeroes
    its encrypted secret payload (same behavior as
    ``archive_vault_credential``). The row persists for audit. Bare DELETE is
    never silently destructive; for the irreversible hard-delete use
    ``POST /v1/vaults/{vault_id}/credentials/{credential_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        credential_id=credential_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    vault_id: str,
    credential_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Delete Credential

     Soft-archive a credential (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the credential from default lists, and zeroes
    its encrypted secret payload (same behavior as
    ``archive_vault_credential``). The row persists for audit. Bare DELETE is
    never silently destructive; for the irreversible hard-delete use
    ``POST /v1/vaults/{vault_id}/credentials/{credential_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | VaultCredential
    """

    return (
        await asyncio_detailed(
            vault_id=vault_id,
            credential_id=credential_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
