from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_event import ListResponseEvent
from ...models.list_session_events_chat_type_type_0 import (
    ListSessionEventsChatTypeType0,
)
from ...models.list_session_events_dir import ListSessionEventsDir
from ...models.list_session_events_kind_type_0 import ListSessionEventsKindType0
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    cursor: None | str | Unset = UNSET,
    after_seq: int | None | Unset = UNSET,
    before_seq: int | None | Unset = UNSET,
    dir_: ListSessionEventsDir | Unset = ListSessionEventsDir.FORWARD,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    error_only: bool | None | Unset = UNSET,
    channel: list[str] | None | Unset = UNSET,
    chat_type: ListSessionEventsChatTypeType0 | None | Unset = UNSET,
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

    json_after_seq: int | None | Unset
    if isinstance(after_seq, Unset):
        json_after_seq = UNSET
    else:
        json_after_seq = after_seq
    params["after_seq"] = json_after_seq

    json_before_seq: int | None | Unset
    if isinstance(before_seq, Unset):
        json_before_seq = UNSET
    else:
        json_before_seq = before_seq
    params["before_seq"] = json_before_seq

    json_dir_: str | Unset = UNSET
    if not isinstance(dir_, Unset):
        json_dir_ = dir_.value

    params["dir"] = json_dir_

    json_kind: None | str | Unset
    if isinstance(kind, Unset):
        json_kind = UNSET
    elif isinstance(kind, ListSessionEventsKindType0):
        json_kind = kind.value
    else:
        json_kind = kind
    params["kind"] = json_kind

    json_error_only: bool | None | Unset
    if isinstance(error_only, Unset):
        json_error_only = UNSET
    else:
        json_error_only = error_only
    params["error_only"] = json_error_only

    json_channel: list[str] | None | Unset
    if isinstance(channel, Unset):
        json_channel = UNSET
    elif isinstance(channel, list):
        json_channel = channel

    else:
        json_channel = channel
    params["channel"] = json_channel

    json_chat_type: None | str | Unset
    if isinstance(chat_type, Unset):
        json_chat_type = UNSET
    elif isinstance(chat_type, ListSessionEventsChatTypeType0):
        json_chat_type = chat_type.value
    else:
        json_chat_type = chat_type
    params["chat_type"] = json_chat_type

    json_limit: int | None | Unset
    if isinstance(limit, Unset):
        json_limit = UNSET
    else:
        json_limit = limit
    params["limit"] = json_limit

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
    cursor: None | str | Unset = UNSET,
    after_seq: int | None | Unset = UNSET,
    before_seq: int | None | Unset = UNSET,
    dir_: ListSessionEventsDir | Unset = ListSessionEventsDir.FORWARD,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    error_only: bool | None | Unset = UNSET,
    channel: list[str] | None | Unset = UNSET,
    chat_type: ListSessionEventsChatTypeType0 | None | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseEvent]:
    r"""List Events

     List a session's events by sequence number.

    First page: ``?dir=forward|backward`` (default forward) + optional
    exclusive ``?after_seq=`` / ``?before_seq=`` window bounds, ``?kind=`` /
    ``?error_only=`` / ``?channel=`` (repeatable, OR) /
    ``?chat_type=dm|group`` + ``?limit=``. Subsequent pages:
    ``?cursor=<next_cursor>`` — the token carries direction and filters, so no
    other params are accepted alongside it. ``forward`` walks oldest→newest;
    ``backward`` loads the newest-first tail and pages into the past.

    Channel/chat_type filter (#1613): ``?channel=C`` returns ONLY events whose
    resolved ``channel == C`` (including the outbound tool-RESULT rows for that
    channel); multiple ``?channel=`` are OR'd. ``?chat_type=`` post-filters on
    the channel address (UUID/numeric ⇒ dm, base64/negative ⇒ group). The
    response includes ``channel`` + ``orig_channel`` on each item.

    Audit readers should persist a processed-through sequence watermark, resume
    with ``after_seq``, and assert that returned sequence numbers provide gapless
    coverage of the requested frozen window as their client-side completeness gate.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"session reset\"
    — it only means no events match this page (e.g. a forward read past the
    current tail, or back-to-back polls racing the writer under load). Page by
    ``seq`` and treat an empty page as \"nothing new yet,\" not as a cleared log.

    Schema (#1140): each item is a session event ``{kind, data, seq}``
    (``kind`` ∈ message/lifecycle/span/interrupt) — a DIFFERENT shape from a
    *run* event (``{type, payload, seq}`` on ``/v1/runs/{id}/events``). See
    ``docs/reference/run-observability.md``.

    Args:
        session_id (str):
        cursor (None | str | Unset):
        after_seq (int | None | Unset):
        before_seq (int | None | Unset):
        dir_ (ListSessionEventsDir | Unset):  Default: ListSessionEventsDir.FORWARD.
        kind (ListSessionEventsKindType0 | None | Unset):
        error_only (bool | None | Unset):
        channel (list[str] | None | Unset):
        chat_type (ListSessionEventsChatTypeType0 | None | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseEvent]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        cursor=cursor,
        after_seq=after_seq,
        before_seq=before_seq,
        dir_=dir_,
        kind=kind,
        error_only=error_only,
        channel=channel,
        chat_type=chat_type,
        limit=limit,
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
    cursor: None | str | Unset = UNSET,
    after_seq: int | None | Unset = UNSET,
    before_seq: int | None | Unset = UNSET,
    dir_: ListSessionEventsDir | Unset = ListSessionEventsDir.FORWARD,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    error_only: bool | None | Unset = UNSET,
    channel: list[str] | None | Unset = UNSET,
    chat_type: ListSessionEventsChatTypeType0 | None | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseEvent | None:
    r"""List Events

     List a session's events by sequence number.

    First page: ``?dir=forward|backward`` (default forward) + optional
    exclusive ``?after_seq=`` / ``?before_seq=`` window bounds, ``?kind=`` /
    ``?error_only=`` / ``?channel=`` (repeatable, OR) /
    ``?chat_type=dm|group`` + ``?limit=``. Subsequent pages:
    ``?cursor=<next_cursor>`` — the token carries direction and filters, so no
    other params are accepted alongside it. ``forward`` walks oldest→newest;
    ``backward`` loads the newest-first tail and pages into the past.

    Channel/chat_type filter (#1613): ``?channel=C`` returns ONLY events whose
    resolved ``channel == C`` (including the outbound tool-RESULT rows for that
    channel); multiple ``?channel=`` are OR'd. ``?chat_type=`` post-filters on
    the channel address (UUID/numeric ⇒ dm, base64/negative ⇒ group). The
    response includes ``channel`` + ``orig_channel`` on each item.

    Audit readers should persist a processed-through sequence watermark, resume
    with ``after_seq``, and assert that returned sequence numbers provide gapless
    coverage of the requested frozen window as their client-side completeness gate.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"session reset\"
    — it only means no events match this page (e.g. a forward read past the
    current tail, or back-to-back polls racing the writer under load). Page by
    ``seq`` and treat an empty page as \"nothing new yet,\" not as a cleared log.

    Schema (#1140): each item is a session event ``{kind, data, seq}``
    (``kind`` ∈ message/lifecycle/span/interrupt) — a DIFFERENT shape from a
    *run* event (``{type, payload, seq}`` on ``/v1/runs/{id}/events``). See
    ``docs/reference/run-observability.md``.

    Args:
        session_id (str):
        cursor (None | str | Unset):
        after_seq (int | None | Unset):
        before_seq (int | None | Unset):
        dir_ (ListSessionEventsDir | Unset):  Default: ListSessionEventsDir.FORWARD.
        kind (ListSessionEventsKindType0 | None | Unset):
        error_only (bool | None | Unset):
        channel (list[str] | None | Unset):
        chat_type (ListSessionEventsChatTypeType0 | None | Unset):
        limit (int | None | Unset):
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
        cursor=cursor,
        after_seq=after_seq,
        before_seq=before_seq,
        dir_=dir_,
        kind=kind,
        error_only=error_only,
        channel=channel,
        chat_type=chat_type,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    after_seq: int | None | Unset = UNSET,
    before_seq: int | None | Unset = UNSET,
    dir_: ListSessionEventsDir | Unset = ListSessionEventsDir.FORWARD,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    error_only: bool | None | Unset = UNSET,
    channel: list[str] | None | Unset = UNSET,
    chat_type: ListSessionEventsChatTypeType0 | None | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseEvent]:
    r"""List Events

     List a session's events by sequence number.

    First page: ``?dir=forward|backward`` (default forward) + optional
    exclusive ``?after_seq=`` / ``?before_seq=`` window bounds, ``?kind=`` /
    ``?error_only=`` / ``?channel=`` (repeatable, OR) /
    ``?chat_type=dm|group`` + ``?limit=``. Subsequent pages:
    ``?cursor=<next_cursor>`` — the token carries direction and filters, so no
    other params are accepted alongside it. ``forward`` walks oldest→newest;
    ``backward`` loads the newest-first tail and pages into the past.

    Channel/chat_type filter (#1613): ``?channel=C`` returns ONLY events whose
    resolved ``channel == C`` (including the outbound tool-RESULT rows for that
    channel); multiple ``?channel=`` are OR'd. ``?chat_type=`` post-filters on
    the channel address (UUID/numeric ⇒ dm, base64/negative ⇒ group). The
    response includes ``channel`` + ``orig_channel`` on each item.

    Audit readers should persist a processed-through sequence watermark, resume
    with ``after_seq``, and assert that returned sequence numbers provide gapless
    coverage of the requested frozen window as their client-side completeness gate.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"session reset\"
    — it only means no events match this page (e.g. a forward read past the
    current tail, or back-to-back polls racing the writer under load). Page by
    ``seq`` and treat an empty page as \"nothing new yet,\" not as a cleared log.

    Schema (#1140): each item is a session event ``{kind, data, seq}``
    (``kind`` ∈ message/lifecycle/span/interrupt) — a DIFFERENT shape from a
    *run* event (``{type, payload, seq}`` on ``/v1/runs/{id}/events``). See
    ``docs/reference/run-observability.md``.

    Args:
        session_id (str):
        cursor (None | str | Unset):
        after_seq (int | None | Unset):
        before_seq (int | None | Unset):
        dir_ (ListSessionEventsDir | Unset):  Default: ListSessionEventsDir.FORWARD.
        kind (ListSessionEventsKindType0 | None | Unset):
        error_only (bool | None | Unset):
        channel (list[str] | None | Unset):
        chat_type (ListSessionEventsChatTypeType0 | None | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseEvent]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        cursor=cursor,
        after_seq=after_seq,
        before_seq=before_seq,
        dir_=dir_,
        kind=kind,
        error_only=error_only,
        channel=channel,
        chat_type=chat_type,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    after_seq: int | None | Unset = UNSET,
    before_seq: int | None | Unset = UNSET,
    dir_: ListSessionEventsDir | Unset = ListSessionEventsDir.FORWARD,
    kind: ListSessionEventsKindType0 | None | Unset = UNSET,
    error_only: bool | None | Unset = UNSET,
    channel: list[str] | None | Unset = UNSET,
    chat_type: ListSessionEventsChatTypeType0 | None | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseEvent | None:
    r"""List Events

     List a session's events by sequence number.

    First page: ``?dir=forward|backward`` (default forward) + optional
    exclusive ``?after_seq=`` / ``?before_seq=`` window bounds, ``?kind=`` /
    ``?error_only=`` / ``?channel=`` (repeatable, OR) /
    ``?chat_type=dm|group`` + ``?limit=``. Subsequent pages:
    ``?cursor=<next_cursor>`` — the token carries direction and filters, so no
    other params are accepted alongside it. ``forward`` walks oldest→newest;
    ``backward`` loads the newest-first tail and pages into the past.

    Channel/chat_type filter (#1613): ``?channel=C`` returns ONLY events whose
    resolved ``channel == C`` (including the outbound tool-RESULT rows for that
    channel); multiple ``?channel=`` are OR'd. ``?chat_type=`` post-filters on
    the channel address (UUID/numeric ⇒ dm, base64/negative ⇒ group). The
    response includes ``channel`` + ``orig_channel`` on each item.

    Audit readers should persist a processed-through sequence watermark, resume
    with ``after_seq``, and assert that returned sequence numbers provide gapless
    coverage of the requested frozen window as their client-side completeness gate.

    Transient-empty (#1140): an empty ``items`` list is NOT a \"session reset\"
    — it only means no events match this page (e.g. a forward read past the
    current tail, or back-to-back polls racing the writer under load). Page by
    ``seq`` and treat an empty page as \"nothing new yet,\" not as a cleared log.

    Schema (#1140): each item is a session event ``{kind, data, seq}``
    (``kind`` ∈ message/lifecycle/span/interrupt) — a DIFFERENT shape from a
    *run* event (``{type, payload, seq}`` on ``/v1/runs/{id}/events``). See
    ``docs/reference/run-observability.md``.

    Args:
        session_id (str):
        cursor (None | str | Unset):
        after_seq (int | None | Unset):
        before_seq (int | None | Unset):
        dir_ (ListSessionEventsDir | Unset):  Default: ListSessionEventsDir.FORWARD.
        kind (ListSessionEventsKindType0 | None | Unset):
        error_only (bool | None | Unset):
        channel (list[str] | None | Unset):
        chat_type (ListSessionEventsChatTypeType0 | None | Unset):
        limit (int | None | Unset):
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
            cursor=cursor,
            after_seq=after_seq,
            before_seq=before_seq,
            dir_=dir_,
            kind=kind,
            error_only=error_only,
            channel=channel,
            chat_type=chat_type,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
