from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_event import ListResponseEvent
from ...models.list_session_events_kind_type_0 import ListSessionEventsKindType0
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    after_seq: int | Unset = 0,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    limit: int | Unset = 200,
    error_only: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["after_seq"] = after_seq

    json_kind: None | str | Unset
    if isinstance(kind, Unset):
        json_kind = UNSET
    elif isinstance(kind, ListSessionEventsKindType0):
        json_kind = kind.value
    else:
        json_kind = kind
    params["kind"] = json_kind

    params["limit"] = limit

    params["error_only"] = error_only

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/events".format(
            session_id=quote(str(session_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseEvent | None:
    if response.status_code == 200:
        response_200 = ListResponseEvent.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseEvent]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    limit: int | Unset = 200,
    error_only: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseEvent]:
    """List Events

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        kind (ListSessionEventsKindType0 | None | Unset):
        limit (int | Unset):  Default: 200.
        error_only (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseEvent]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        after_seq=after_seq,
        kind=kind,
        limit=limit,
        error_only=error_only,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    limit: int | Unset = 200,
    error_only: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseEvent | None:
    """List Events

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        kind (ListSessionEventsKindType0 | None | Unset):
        limit (int | Unset):  Default: 200.
        error_only (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseEvent
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        after_seq=after_seq,
        kind=kind,
        limit=limit,
        error_only=error_only,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    limit: int | Unset = 200,
    error_only: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseEvent]:
    """List Events

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        kind (ListSessionEventsKindType0 | None | Unset):
        limit (int | Unset):  Default: 200.
        error_only (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseEvent]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        after_seq=after_seq,
        kind=kind,
        limit=limit,
        error_only=error_only,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after_seq: int | Unset = 0,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    limit: int | Unset = 200,
    error_only: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseEvent | None:
    """List Events

    Args:
        session_id (str):
        after_seq (int | Unset):  Default: 0.
        kind (ListSessionEventsKindType0 | None | Unset):
        limit (int | Unset):  Default: 200.
        error_only (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseEvent
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            after_seq=after_seq,
            kind=kind,
            limit=limit,
            error_only=error_only,
            authorization=authorization,
        )
    ).parsed
