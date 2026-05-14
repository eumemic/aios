from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.bootstrap_request import BootstrapRequest
from ...models.bootstrap_response import BootstrapResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/accounts/bootstrap",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> BootstrapResponse | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = BootstrapResponse.from_dict(response.json())

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
) -> Response[BootstrapResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[BootstrapResponse | HTTPValidationError]:
    """Bootstrap

     One-shot endpoint that mints the root account and its first API key.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value — a fresh
    deployment must explicitly opt in to bootstrap by setting the env.

    Once a non-archived root account exists, the endpoint is 404
    regardless of token validity. The ``accounts_one_active_root``
    partial unique index in migration 0040 enforces the invariant at
    the DB layer too — the 404 here is the friendly upstream answer.

    The ``plaintext_key`` field of the response is the *only* time the
    operator key is returned in plaintext. After this call, every
    subsequent use of that key authenticates against the stored
    ``sha256`` hash.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest): Body for ``POST /v1/accounts/bootstrap``.

            Only the human-readable display_name is required at bootstrap time;
            metadata can be added later via a PATCH endpoint (lands in PR 6).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BootstrapResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> BootstrapResponse | HTTPValidationError | None:
    """Bootstrap

     One-shot endpoint that mints the root account and its first API key.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value — a fresh
    deployment must explicitly opt in to bootstrap by setting the env.

    Once a non-archived root account exists, the endpoint is 404
    regardless of token validity. The ``accounts_one_active_root``
    partial unique index in migration 0040 enforces the invariant at
    the DB layer too — the 404 here is the friendly upstream answer.

    The ``plaintext_key`` field of the response is the *only* time the
    operator key is returned in plaintext. After this call, every
    subsequent use of that key authenticates against the stored
    ``sha256`` hash.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest): Body for ``POST /v1/accounts/bootstrap``.

            Only the human-readable display_name is required at bootstrap time;
            metadata can be added later via a PATCH endpoint (lands in PR 6).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BootstrapResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[BootstrapResponse | HTTPValidationError]:
    """Bootstrap

     One-shot endpoint that mints the root account and its first API key.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value — a fresh
    deployment must explicitly opt in to bootstrap by setting the env.

    Once a non-archived root account exists, the endpoint is 404
    regardless of token validity. The ``accounts_one_active_root``
    partial unique index in migration 0040 enforces the invariant at
    the DB layer too — the 404 here is the friendly upstream answer.

    The ``plaintext_key`` field of the response is the *only* time the
    operator key is returned in plaintext. After this call, every
    subsequent use of that key authenticates against the stored
    ``sha256`` hash.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest): Body for ``POST /v1/accounts/bootstrap``.

            Only the human-readable display_name is required at bootstrap time;
            metadata can be added later via a PATCH endpoint (lands in PR 6).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BootstrapResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> BootstrapResponse | HTTPValidationError | None:
    """Bootstrap

     One-shot endpoint that mints the root account and its first API key.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value — a fresh
    deployment must explicitly opt in to bootstrap by setting the env.

    Once a non-archived root account exists, the endpoint is 404
    regardless of token validity. The ``accounts_one_active_root``
    partial unique index in migration 0040 enforces the invariant at
    the DB layer too — the 404 here is the friendly upstream answer.

    The ``plaintext_key`` field of the response is the *only* time the
    operator key is returned in plaintext. After this call, every
    subsequent use of that key authenticates against the stored
    ``sha256`` hash.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest): Body for ``POST /v1/accounts/bootstrap``.

            Only the human-readable display_name is required at bootstrap time;
            metadata can be added later via a PATCH endpoint (lands in PR 6).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BootstrapResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
