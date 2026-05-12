from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_tool_result_request import ConnectorToolResultRequest
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ConnectorToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/tool-results",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = response.json()
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
) -> Response[Any | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Tool Result

     Submit a custom tool result from a connector container.

    Authorization: the session must be bound to the caller's connection
    (single_session attach, per_chat origin, or operator-bound chat).
    Otherwise → 403.

    Args:
        authorization (None | str | Unset):
        body (ConnectorToolResultRequest): Body for ``POST /v1/connectors/tool-results``.

            Mirrors the operator-facing :class:`ToolResultRequest` but adds
            ``session_id`` since connector tokens aren't path-scoped to a
            session.  The handler validates the session is bound to the
            caller's connection (preventing a connector from posting results
            for sessions outside its scope).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Tool Result

     Submit a custom tool result from a connector container.

    Authorization: the session must be bound to the caller's connection
    (single_session attach, per_chat origin, or operator-bound chat).
    Otherwise → 403.

    Args:
        authorization (None | str | Unset):
        body (ConnectorToolResultRequest): Body for ``POST /v1/connectors/tool-results``.

            Mirrors the operator-facing :class:`ToolResultRequest` but adds
            ``session_id`` since connector tokens aren't path-scoped to a
            session.  The handler validates the session is bound to the
            caller's connection (preventing a connector from posting results
            for sessions outside its scope).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Tool Result

     Submit a custom tool result from a connector container.

    Authorization: the session must be bound to the caller's connection
    (single_session attach, per_chat origin, or operator-bound chat).
    Otherwise → 403.

    Args:
        authorization (None | str | Unset):
        body (ConnectorToolResultRequest): Body for ``POST /v1/connectors/tool-results``.

            Mirrors the operator-facing :class:`ToolResultRequest` but adds
            ``session_id`` since connector tokens aren't path-scoped to a
            session.  The handler validates the session is bound to the
            caller's connection (preventing a connector from posting results
            for sessions outside its scope).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorToolResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Tool Result

     Submit a custom tool result from a connector container.

    Authorization: the session must be bound to the caller's connection
    (single_session attach, per_chat origin, or operator-bound chat).
    Otherwise → 403.

    Args:
        authorization (None | str | Unset):
        body (ConnectorToolResultRequest): Body for ``POST /v1/connectors/tool-results``.

            Mirrors the operator-facing :class:`ToolResultRequest` but adds
            ``session_id`` since connector tokens aren't path-scoped to a
            session.  The handler validates the session is bound to the
            caller's connection (preventing a connector from posting results
            for sessions outside its scope).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
