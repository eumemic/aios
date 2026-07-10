from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.model_provider import ModelProvider
from ...models.model_provider_update import ModelProviderUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    model_provider_id: str,
    *,
    body: ModelProviderUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/model-providers/{model_provider_id}".format(
            model_provider_id=quote(str(model_provider_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ModelProvider | None:
    if response.status_code == 200:
        response_200 = ModelProvider.from_dict(response.json())

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
) -> Response[HTTPValidationError | ModelProvider]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    model_provider_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ModelProviderUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ModelProvider]:
    """Update

     Rotate the key and/or edit ``api_base``.

    ``api_key`` omitted → keep the existing key (there is no way to clear it
    back to unset — archive and recreate instead). ``api_base`` omitted →
    keep; explicit ``null`` → clear.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):
        body (ModelProviderUpdate): Request body for ``PUT /v1/model-providers/{id}``.

            ``api_key`` omitted → keep the existing key (rotation is opt-in via an
            explicit value; there is no way to clear it back to unset in v1 — archive
            and recreate instead). ``api_base`` omitted → keep; explicit ``null`` →
            clear (checked via ``model_fields_set``, not a sentinel default, since
            ``None`` is itself a valid target value).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ModelProvider]
    """

    kwargs = _get_kwargs(
        model_provider_id=model_provider_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    model_provider_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ModelProviderUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ModelProvider | None:
    """Update

     Rotate the key and/or edit ``api_base``.

    ``api_key`` omitted → keep the existing key (there is no way to clear it
    back to unset — archive and recreate instead). ``api_base`` omitted →
    keep; explicit ``null`` → clear.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):
        body (ModelProviderUpdate): Request body for ``PUT /v1/model-providers/{id}``.

            ``api_key`` omitted → keep the existing key (rotation is opt-in via an
            explicit value; there is no way to clear it back to unset in v1 — archive
            and recreate instead). ``api_base`` omitted → keep; explicit ``null`` →
            clear (checked via ``model_fields_set``, not a sentinel default, since
            ``None`` is itself a valid target value).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ModelProvider
    """

    return sync_detailed(
        model_provider_id=model_provider_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    model_provider_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ModelProviderUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ModelProvider]:
    """Update

     Rotate the key and/or edit ``api_base``.

    ``api_key`` omitted → keep the existing key (there is no way to clear it
    back to unset — archive and recreate instead). ``api_base`` omitted →
    keep; explicit ``null`` → clear.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):
        body (ModelProviderUpdate): Request body for ``PUT /v1/model-providers/{id}``.

            ``api_key`` omitted → keep the existing key (rotation is opt-in via an
            explicit value; there is no way to clear it back to unset in v1 — archive
            and recreate instead). ``api_base`` omitted → keep; explicit ``null`` →
            clear (checked via ``model_fields_set``, not a sentinel default, since
            ``None`` is itself a valid target value).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ModelProvider]
    """

    kwargs = _get_kwargs(
        model_provider_id=model_provider_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    model_provider_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ModelProviderUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ModelProvider | None:
    """Update

     Rotate the key and/or edit ``api_base``.

    ``api_key`` omitted → keep the existing key (there is no way to clear it
    back to unset — archive and recreate instead). ``api_base`` omitted →
    keep; explicit ``null`` → clear.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):
        body (ModelProviderUpdate): Request body for ``PUT /v1/model-providers/{id}``.

            ``api_key`` omitted → keep the existing key (rotation is opt-in via an
            explicit value; there is no way to clear it back to unset in v1 — archive
            and recreate instead). ``api_base`` omitted → keep; explicit ``null`` →
            clear (checked via ``model_fields_set``, not a sentinel default, since
            ``None`` is itself a valid target value).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ModelProvider
    """

    return (
        await asyncio_detailed(
            model_provider_id=model_provider_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
