from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_wf_run_event import ListResponseWfRunEvent
from ...types import UNSET, Response, Unset


def _get_kwargs(
    run_id: str,
    *,
    cursor: None | str | Unset = UNSET,
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

    json_limit: int | None | Unset
    if isinstance(limit, Unset):
        json_limit = UNSET
    else:
        json_limit = limit
    params["limit"] = json_limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/runs/{run_id}/events".format(
            run_id=quote(str(run_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseWfRunEvent | None:
    if response.status_code == 200:
        response_200 = ListResponseWfRunEvent.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseWfRunEvent]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseWfRunEvent]:
    r"""List Run Events

     A run's journal by sequence (oldest first). First page: optional ``limit``;
    subsequent pages: ``?cursor=<next_cursor>``.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"run reset\" — it
    only means no journal rows past this ``seq`` yet. Page by ``seq`` and treat
    an empty page as \"nothing new yet.\"

    Schema (#1140): each item is a *run* event ``{type, payload, seq}`` — a
    DIFFERENT shape from a child-*session* event (``{kind, data}`` on
    ``/v1/sessions/{id}/events``). See ``docs/reference/run-observability.md``.

    Args:
        run_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseWfRunEvent]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        cursor=cursor,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseWfRunEvent | None:
    r"""List Run Events

     A run's journal by sequence (oldest first). First page: optional ``limit``;
    subsequent pages: ``?cursor=<next_cursor>``.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"run reset\" — it
    only means no journal rows past this ``seq`` yet. Page by ``seq`` and treat
    an empty page as \"nothing new yet.\"

    Schema (#1140): each item is a *run* event ``{type, payload, seq}`` — a
    DIFFERENT shape from a child-*session* event (``{kind, data}`` on
    ``/v1/sessions/{id}/events``). See ``docs/reference/run-observability.md``.

    Args:
        run_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseWfRunEvent
    """

    return sync_detailed(
        run_id=run_id,
        client=client,
        cursor=cursor,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseWfRunEvent]:
    r"""List Run Events

     A run's journal by sequence (oldest first). First page: optional ``limit``;
    subsequent pages: ``?cursor=<next_cursor>``.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"run reset\" — it
    only means no journal rows past this ``seq`` yet. Page by ``seq`` and treat
    an empty page as \"nothing new yet.\"

    Schema (#1140): each item is a *run* event ``{type, payload, seq}`` — a
    DIFFERENT shape from a child-*session* event (``{kind, data}`` on
    ``/v1/sessions/{id}/events``). See ``docs/reference/run-observability.md``.

    Args:
        run_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseWfRunEvent]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        cursor=cursor,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseWfRunEvent | None:
    r"""List Run Events

     A run's journal by sequence (oldest first). First page: optional ``limit``;
    subsequent pages: ``?cursor=<next_cursor>``.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"run reset\" — it
    only means no journal rows past this ``seq`` yet. Page by ``seq`` and treat
    an empty page as \"nothing new yet.\"

    Schema (#1140): each item is a *run* event ``{type, payload, seq}`` — a
    DIFFERENT shape from a child-*session* event (``{kind, data}`` on
    ``/v1/sessions/{id}/events``). See ``docs/reference/run-observability.md``.

    Args:
        run_id (str):
        cursor (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseWfRunEvent
    """

    return (
        await asyncio_detailed(
            run_id=run_id,
            client=client,
            cursor=cursor,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
