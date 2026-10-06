from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_trigger_run_echo import ListResponseTriggerRunEcho
from ...types import UNSET, Response, Unset


def _get_kwargs(
    name: str,
    *,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/triggers/{name}/runs".format(
            name=quote(str(name), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseTriggerRunEcho | None:
    if response.status_code == 200:
        response_200 = ListResponseTriggerRunEcho.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseTriggerRunEcho]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseTriggerRunEcho]:
    """List Operator Trigger Runs

     List an operator trigger's fires (the per-fire audit), newest first. Keyed by
    name, so a deleted trigger's history stays readable until retention prunes it.

    Args:
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseTriggerRunEcho]
    """

    kwargs = _get_kwargs(
        name=name,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseTriggerRunEcho | None:
    """List Operator Trigger Runs

     List an operator trigger's fires (the per-fire audit), newest first. Keyed by
    name, so a deleted trigger's history stays readable until retention prunes it.

    Args:
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseTriggerRunEcho
    """

    return sync_detailed(
        name=name,
        client=client,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseTriggerRunEcho]:
    """List Operator Trigger Runs

     List an operator trigger's fires (the per-fire audit), newest first. Keyed by
    name, so a deleted trigger's history stays readable until retention prunes it.

    Args:
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseTriggerRunEcho]
    """

    kwargs = _get_kwargs(
        name=name,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseTriggerRunEcho | None:
    """List Operator Trigger Runs

     List an operator trigger's fires (the per-fire audit), newest first. Keyed by
    name, so a deleted trigger's history stays readable until retention prunes it.

    Args:
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseTriggerRunEcho
    """

    return (
        await asyncio_detailed(
            name=name,
            client=client,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
