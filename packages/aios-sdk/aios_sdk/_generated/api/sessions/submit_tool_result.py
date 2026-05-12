from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.event import Event
from ...models.http_validation_error import HTTPValidationError
from ...models.tool_result_request import ToolResultRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: ToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/tool-results".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Event | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = Event.from_dict(response.json())

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
) -> Response[Event | HTTPValidationError]:
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
    body: ToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Event | HTTPValidationError]:
    """Submit Tool Result

     Submit a custom tool result. Appends a tool-role message and wakes the session.

    Stamps the tool's ``name`` into the event data by looking it up on the
    parent assistant's ``tool_calls`` array — same source the harness uses
    for built-in/MCP results — so the derived ``tool_name`` column stays
    populated for custom tools too (issue #133).  Returns 404 when the
    ``tool_call_id`` has no matching parent assistant tool call, since a
    result with no parent is a client bug that would leave an orphan row.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolResultRequest): Request body for ``POST /v1/sessions/{id}/tool-results``.

            ``content`` accepts either a plain string OR a multimodal content
            array shaped per the OpenAI chat-completions tool-result format
            (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url",
            "image_url": {"url": "..."}}]``).  Built-in tools have always
            produced multimodal results; this widening lets external clients
            (#301 — connectors as HTTP clients) do the same when posting
            custom-tool results.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Event | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
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
    body: ToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Event | HTTPValidationError | None:
    """Submit Tool Result

     Submit a custom tool result. Appends a tool-role message and wakes the session.

    Stamps the tool's ``name`` into the event data by looking it up on the
    parent assistant's ``tool_calls`` array — same source the harness uses
    for built-in/MCP results — so the derived ``tool_name`` column stays
    populated for custom tools too (issue #133).  Returns 404 when the
    ``tool_call_id`` has no matching parent assistant tool call, since a
    result with no parent is a client bug that would leave an orphan row.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolResultRequest): Request body for ``POST /v1/sessions/{id}/tool-results``.

            ``content`` accepts either a plain string OR a multimodal content
            array shaped per the OpenAI chat-completions tool-result format
            (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url",
            "image_url": {"url": "..."}}]``).  Built-in tools have always
            produced multimodal results; this widening lets external clients
            (#301 — connectors as HTTP clients) do the same when posting
            custom-tool results.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Event | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Event | HTTPValidationError]:
    """Submit Tool Result

     Submit a custom tool result. Appends a tool-role message and wakes the session.

    Stamps the tool's ``name`` into the event data by looking it up on the
    parent assistant's ``tool_calls`` array — same source the harness uses
    for built-in/MCP results — so the derived ``tool_name`` column stays
    populated for custom tools too (issue #133).  Returns 404 when the
    ``tool_call_id`` has no matching parent assistant tool call, since a
    result with no parent is a client bug that would leave an orphan row.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolResultRequest): Request body for ``POST /v1/sessions/{id}/tool-results``.

            ``content`` accepts either a plain string OR a multimodal content
            array shaped per the OpenAI chat-completions tool-result format
            (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url",
            "image_url": {"url": "..."}}]``).  Built-in tools have always
            produced multimodal results; this widening lets external clients
            (#301 — connectors as HTTP clients) do the same when posting
            custom-tool results.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Event | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Event | HTTPValidationError | None:
    """Submit Tool Result

     Submit a custom tool result. Appends a tool-role message and wakes the session.

    Stamps the tool's ``name`` into the event data by looking it up on the
    parent assistant's ``tool_calls`` array — same source the harness uses
    for built-in/MCP results — so the derived ``tool_name`` column stays
    populated for custom tools too (issue #133).  Returns 404 when the
    ``tool_call_id`` has no matching parent assistant tool call, since a
    result with no parent is a client bug that would leave an orphan row.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolResultRequest): Request body for ``POST /v1/sessions/{id}/tool-results``.

            ``content`` accepts either a plain string OR a multimodal content
            array shaped per the OpenAI chat-completions tool-result format
            (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url",
            "image_url": {"url": "..."}}]``).  Built-in tools have always
            produced multimodal results; this widening lets external clients
            (#301 — connectors as HTTP clients) do the same when posting
            custom-tool results.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Event | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
