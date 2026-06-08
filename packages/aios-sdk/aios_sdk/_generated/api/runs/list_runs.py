from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_wf_run import ListResponseWfRun
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    cursor: None | str | Unset = UNSET,
    workflow_id: None | str | Unset = UNSET,
    status: None | str | Unset = UNSET,
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

    json_workflow_id: None | str | Unset
    if isinstance(workflow_id, Unset):
        json_workflow_id = UNSET
    else:
        json_workflow_id = workflow_id
    params["workflow_id"] = json_workflow_id

    json_status: None | str | Unset
    if isinstance(status, Unset):
        json_status = UNSET
    else:
        json_status = status
    params["status"] = json_status

    json_limit: int | None | Unset
    if isinstance(limit, Unset):
        json_limit = UNSET
    else:
        json_limit = limit
    params["limit"] = json_limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/runs",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseWfRun | None:
    if response.status_code == 200:
        response_200 = ListResponseWfRun.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseWfRun]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    workflow_id: None | str | Unset = UNSET,
    status: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseWfRun]:
    """List Runs

     List the account's runs, newest first. First page: optional ``workflow_id`` /
    ``status`` filters + ``limit``; subsequent pages: ``?cursor=<next_cursor>``.

    Args:
        cursor (None | str | Unset):
        workflow_id (None | str | Unset):
        status (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseWfRun]
    """

    kwargs = _get_kwargs(
        cursor=cursor,
        workflow_id=workflow_id,
        status=status,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    workflow_id: None | str | Unset = UNSET,
    status: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseWfRun | None:
    """List Runs

     List the account's runs, newest first. First page: optional ``workflow_id`` /
    ``status`` filters + ``limit``; subsequent pages: ``?cursor=<next_cursor>``.

    Args:
        cursor (None | str | Unset):
        workflow_id (None | str | Unset):
        status (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseWfRun
    """

    return sync_detailed(
        client=client,
        cursor=cursor,
        workflow_id=workflow_id,
        status=status,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    workflow_id: None | str | Unset = UNSET,
    status: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseWfRun]:
    """List Runs

     List the account's runs, newest first. First page: optional ``workflow_id`` /
    ``status`` filters + ``limit``; subsequent pages: ``?cursor=<next_cursor>``.

    Args:
        cursor (None | str | Unset):
        workflow_id (None | str | Unset):
        status (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseWfRun]
    """

    kwargs = _get_kwargs(
        cursor=cursor,
        workflow_id=workflow_id,
        status=status,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    workflow_id: None | str | Unset = UNSET,
    status: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseWfRun | None:
    """List Runs

     List the account's runs, newest first. First page: optional ``workflow_id`` /
    ``status`` filters + ``limit``; subsequent pages: ``?cursor=<next_cursor>``.

    Args:
        cursor (None | str | Unset):
        workflow_id (None | str | Unset):
        status (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseWfRun
    """

    return (
        await asyncio_detailed(
            client=client,
            cursor=cursor,
            workflow_id=workflow_id,
            status=status,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
