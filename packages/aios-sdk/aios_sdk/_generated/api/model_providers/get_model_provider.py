from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.model_provider import ModelProvider
from ...types import UNSET, Response, Unset


def _get_kwargs(
    model_provider_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/model-providers/{model_provider_id}".format(
            model_provider_id=quote(str(model_provider_id), safe=""),
        ),
    }

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
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ModelProvider]:
    """Get

     Fetch one model-provider config by id. ``api_key`` is never returned.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ModelProvider]
    """

    kwargs = _get_kwargs(
        model_provider_id=model_provider_id,
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
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ModelProvider | None:
    """Get

     Fetch one model-provider config by id. ``api_key`` is never returned.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ModelProvider
    """

    return sync_detailed(
        model_provider_id=model_provider_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    model_provider_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ModelProvider]:
    """Get

     Fetch one model-provider config by id. ``api_key`` is never returned.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ModelProvider]
    """

    kwargs = _get_kwargs(
        model_provider_id=model_provider_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    model_provider_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ModelProvider | None:
    """Get

     Fetch one model-provider config by id. ``api_key`` is never returned.

    Args:
        model_provider_id (str):
        authorization (None | str | Unset):

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
            authorization=authorization,
        )
    ).parsed
