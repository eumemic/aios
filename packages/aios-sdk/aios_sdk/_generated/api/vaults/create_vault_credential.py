from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.vault_credential import VaultCredential
from ...models.vault_credential_create import VaultCredentialCreate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    *,
    body: VaultCredentialCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/vaults/{vault_id}/credentials".format(
            vault_id=quote(str(vault_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | VaultCredential | None:
    if response.status_code == 201:
        response_201 = VaultCredential.from_dict(response.json())

        return response_201

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
    *,
    client: AuthenticatedClient | Client,
    body: VaultCredentialCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Create Credential

     Add a credential to a vault. Secrets are encrypted at rest via the CryptoBox.

    Validates required fields per ``auth_type``: ``oauth2_refresh`` requires
    ``access_token`` (plus the refresh fields needed for rotation);
    ``bearer_header`` requires ``token``; ``basic`` requires ``username``
    and ``password``. Caps at 20 active credentials per vault. The
    ``target_url`` is immutable after creation — to retarget a credential,
    archive the existing one and create a new credential at the new URL.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialCreate): Request body for ``POST /v1/vaults/{vault_id}/credentials``.

            All secret fields are write-only. The ``target_url`` is immutable
            after creation. The service layer validates required fields per
            ``auth_type``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        body=body,
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
    body: VaultCredentialCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Create Credential

     Add a credential to a vault. Secrets are encrypted at rest via the CryptoBox.

    Validates required fields per ``auth_type``: ``oauth2_refresh`` requires
    ``access_token`` (plus the refresh fields needed for rotation);
    ``bearer_header`` requires ``token``; ``basic`` requires ``username``
    and ``password``. Caps at 20 active credentials per vault. The
    ``target_url`` is immutable after creation — to retarget a credential,
    archive the existing one and create a new credential at the new URL.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialCreate): Request body for ``POST /v1/vaults/{vault_id}/credentials``.

            All secret fields are write-only. The ``target_url`` is immutable
            after creation. The service layer validates required fields per
            ``auth_type``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | VaultCredential
    """

    return sync_detailed(
        vault_id=vault_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: VaultCredentialCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Create Credential

     Add a credential to a vault. Secrets are encrypted at rest via the CryptoBox.

    Validates required fields per ``auth_type``: ``oauth2_refresh`` requires
    ``access_token`` (plus the refresh fields needed for rotation);
    ``bearer_header`` requires ``token``; ``basic`` requires ``username``
    and ``password``. Caps at 20 active credentials per vault. The
    ``target_url`` is immutable after creation — to retarget a credential,
    archive the existing one and create a new credential at the new URL.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialCreate): Request body for ``POST /v1/vaults/{vault_id}/credentials``.

            All secret fields are write-only. The ``target_url`` is immutable
            after creation. The service layer validates required fields per
            ``auth_type``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: VaultCredentialCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Create Credential

     Add a credential to a vault. Secrets are encrypted at rest via the CryptoBox.

    Validates required fields per ``auth_type``: ``oauth2_refresh`` requires
    ``access_token`` (plus the refresh fields needed for rotation);
    ``bearer_header`` requires ``token``; ``basic`` requires ``username``
    and ``password``. Caps at 20 active credentials per vault. The
    ``target_url`` is immutable after creation — to retarget a credential,
    archive the existing one and create a new credential at the new URL.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialCreate): Request body for ``POST /v1/vaults/{vault_id}/credentials``.

            All secret fields are write-only. The ``target_url`` is immutable
            after creation. The service layer validates required fields per
            ``auth_type``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | VaultCredential
    """

    return (
        await asyncio_detailed(
            vault_id=vault_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
