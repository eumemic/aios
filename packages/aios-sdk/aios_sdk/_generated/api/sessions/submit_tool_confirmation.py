from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.event import Event
from ...models.http_validation_error import HTTPValidationError
from ...models.tool_confirmation_request import ToolConfirmationRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: ToolConfirmationRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/tool-confirmations".format(
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
    body: ToolConfirmationRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Event | HTTPValidationError]:
    """Submit Tool Confirmation

     Confirm or deny an ``always_ask`` built-in tool call.

    ``allow`` records a lifecycle event; the worker dispatches the tool on
    its next step.  ``deny`` appends a tool-role error event; the model
    sees the denial message and can adapt.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolConfirmationRequest): Request body for ``POST /v1/sessions/{id}/tool-
            confirmations``.

            Used for built-in tools with ``permission: "always_ask"``. The client
            inspects the pending tool call and either allows it (the worker will
            execute it) or denies it (the model receives an error with the deny
            message).

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
    body: ToolConfirmationRequest,
    authorization: None | str | Unset = UNSET,
) -> Event | HTTPValidationError | None:
    """Submit Tool Confirmation

     Confirm or deny an ``always_ask`` built-in tool call.

    ``allow`` records a lifecycle event; the worker dispatches the tool on
    its next step.  ``deny`` appends a tool-role error event; the model
    sees the denial message and can adapt.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolConfirmationRequest): Request body for ``POST /v1/sessions/{id}/tool-
            confirmations``.

            Used for built-in tools with ``permission: "always_ask"``. The client
            inspects the pending tool call and either allows it (the worker will
            execute it) or denies it (the model receives an error with the deny
            message).

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
    body: ToolConfirmationRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Event | HTTPValidationError]:
    """Submit Tool Confirmation

     Confirm or deny an ``always_ask`` built-in tool call.

    ``allow`` records a lifecycle event; the worker dispatches the tool on
    its next step.  ``deny`` appends a tool-role error event; the model
    sees the denial message and can adapt.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolConfirmationRequest): Request body for ``POST /v1/sessions/{id}/tool-
            confirmations``.

            Used for built-in tools with ``permission: "always_ask"``. The client
            inspects the pending tool call and either allows it (the worker will
            execute it) or denies it (the model receives an error with the deny
            message).

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
    body: ToolConfirmationRequest,
    authorization: None | str | Unset = UNSET,
) -> Event | HTTPValidationError | None:
    """Submit Tool Confirmation

     Confirm or deny an ``always_ask`` built-in tool call.

    ``allow`` records a lifecycle event; the worker dispatches the tool on
    its next step.  ``deny`` appends a tool-role error event; the model
    sees the denial message and can adapt.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ToolConfirmationRequest): Request body for ``POST /v1/sessions/{id}/tool-
            confirmations``.

            Used for built-in tools with ``permission: "always_ask"``. The client
            inspects the pending tool call and either allows it (the worker will
            execute it) or denies it (the model receives an error with the deny
            message).

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
