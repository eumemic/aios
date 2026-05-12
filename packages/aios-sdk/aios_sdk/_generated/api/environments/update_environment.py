from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.environment import Environment
from ...models.environment_update import EnvironmentUpdate
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    env_id: str,
    *,
    body: EnvironmentUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/environments/{env_id}".format(
            env_id=quote(str(env_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Environment | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Environment.from_dict(response.json())

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
) -> Response[Environment | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    env_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: EnvironmentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Environment | HTTPValidationError]:
    """Update

     Update an environment's ``name`` and/or ``config``.

    Sessions resolve the environment config fresh each time their sandbox
    is provisioned (lazily, at the next session step that needs the
    container), so updates take effect for existing sessions on their next
    provision rather than at update time.

    Args:
        env_id (str):
        authorization (None | str | Unset):
        body (EnvironmentUpdate): Request body for ``PUT /v1/environments/{id}``.

            All fields are optional; omitted fields are preserved.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Environment | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        env_id=env_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    env_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: EnvironmentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Environment | HTTPValidationError | None:
    """Update

     Update an environment's ``name`` and/or ``config``.

    Sessions resolve the environment config fresh each time their sandbox
    is provisioned (lazily, at the next session step that needs the
    container), so updates take effect for existing sessions on their next
    provision rather than at update time.

    Args:
        env_id (str):
        authorization (None | str | Unset):
        body (EnvironmentUpdate): Request body for ``PUT /v1/environments/{id}``.

            All fields are optional; omitted fields are preserved.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Environment | HTTPValidationError
    """

    return sync_detailed(
        env_id=env_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    env_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: EnvironmentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Environment | HTTPValidationError]:
    """Update

     Update an environment's ``name`` and/or ``config``.

    Sessions resolve the environment config fresh each time their sandbox
    is provisioned (lazily, at the next session step that needs the
    container), so updates take effect for existing sessions on their next
    provision rather than at update time.

    Args:
        env_id (str):
        authorization (None | str | Unset):
        body (EnvironmentUpdate): Request body for ``PUT /v1/environments/{id}``.

            All fields are optional; omitted fields are preserved.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Environment | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        env_id=env_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    env_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: EnvironmentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Environment | HTTPValidationError | None:
    """Update

     Update an environment's ``name`` and/or ``config``.

    Sessions resolve the environment config fresh each time their sandbox
    is provisioned (lazily, at the next session step that needs the
    container), so updates take effect for existing sessions on their next
    provision rather than at update time.

    Args:
        env_id (str):
        authorization (None | str | Unset):
        body (EnvironmentUpdate): Request body for ``PUT /v1/environments/{id}``.

            All fields are optional; omitted fields are preserved.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Environment | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            env_id=env_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
