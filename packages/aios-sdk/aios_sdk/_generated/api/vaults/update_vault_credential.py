from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.vault_credential import VaultCredential
from ...models.vault_credential_update import VaultCredentialUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    credential_id: str,
    *,
    body: VaultCredentialUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/vaults/{vault_id}/credentials/{credential_id}".format(
            vault_id=quote(str(vault_id), safe=""),
            credential_id=quote(str(credential_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

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
    body: VaultCredentialUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Update Credential

     Update a credential's metadata and/or rotate its auth secrets.

    Omitted secret fields are preserved (decrypt-merge-encrypt cycle on the
    encrypted payload). ``target_url`` and ``auth_type`` are immutable
    and not accepted in the body. To rotate an OAuth refresh token, send
    only the new ``refresh_token`` (and optional ``access_token`` /
    ``expires_at``); other auth fields stay intact.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialUpdate): Request body for ``PUT
            /v1/vaults/{vault_id}/credentials/{id}``.

            ``target_url``, ``secret_name``, ``allowed_hosts``, and ``auth_type`` are
            immutable — not accepted here. Omitted secret fields are preserved
            (decrypt-merge-encrypt).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        credential_id=credential_id,
        body=body,
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
    body: VaultCredentialUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Update Credential

     Update a credential's metadata and/or rotate its auth secrets.

    Omitted secret fields are preserved (decrypt-merge-encrypt cycle on the
    encrypted payload). ``target_url`` and ``auth_type`` are immutable
    and not accepted in the body. To rotate an OAuth refresh token, send
    only the new ``refresh_token`` (and optional ``access_token`` /
    ``expires_at``); other auth fields stay intact.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialUpdate): Request body for ``PUT
            /v1/vaults/{vault_id}/credentials/{id}``.

            ``target_url``, ``secret_name``, ``allowed_hosts``, and ``auth_type`` are
            immutable — not accepted here. Omitted secret fields are preserved
            (decrypt-merge-encrypt).

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
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    vault_id: str,
    credential_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: VaultCredentialUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Update Credential

     Update a credential's metadata and/or rotate its auth secrets.

    Omitted secret fields are preserved (decrypt-merge-encrypt cycle on the
    encrypted payload). ``target_url`` and ``auth_type`` are immutable
    and not accepted in the body. To rotate an OAuth refresh token, send
    only the new ``refresh_token`` (and optional ``access_token`` /
    ``expires_at``); other auth fields stay intact.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialUpdate): Request body for ``PUT
            /v1/vaults/{vault_id}/credentials/{id}``.

            ``target_url``, ``secret_name``, ``allowed_hosts``, and ``auth_type`` are
            immutable — not accepted here. Omitted secret fields are preserved
            (decrypt-merge-encrypt).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        credential_id=credential_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    vault_id: str,
    credential_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: VaultCredentialUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Update Credential

     Update a credential's metadata and/or rotate its auth secrets.

    Omitted secret fields are preserved (decrypt-merge-encrypt cycle on the
    encrypted payload). ``target_url`` and ``auth_type`` are immutable
    and not accepted in the body. To rotate an OAuth refresh token, send
    only the new ``refresh_token`` (and optional ``access_token`` /
    ``expires_at``); other auth fields stay intact.

    Args:
        vault_id (str):
        credential_id (str):
        authorization (None | str | Unset):
        body (VaultCredentialUpdate): Request body for ``PUT
            /v1/vaults/{vault_id}/credentials/{id}``.

            ``target_url``, ``secret_name``, ``allowed_hosts``, and ``auth_type`` are
            immutable — not accepted here. Omitted secret fields are preserved
            (decrypt-merge-encrypt).

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
            body=body,
            authorization=authorization,
        )
    ).parsed
