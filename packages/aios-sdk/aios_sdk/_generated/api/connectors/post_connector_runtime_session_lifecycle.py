from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.post_connector_runtime_session_lifecycle_response_post_connector_runtime_session_lifecycle import (
    PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle,
)
from ...models.runtime_session_lifecycle_request import RuntimeSessionLifecycleRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: RuntimeSessionLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/runtime/session-lifecycle",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    HTTPValidationError
    | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
    | None
):
    if response.status_code == 201:
        response_201 = PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle.from_dict(
            response.json()
        )

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
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: RuntimeSessionLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
]:
    """Post Runtime Session Lifecycle

     Append a ``kind=lifecycle`` event onto **one** session bound to
    ``body.connection_id`` (#1261), optionally waking it.

    The per-session-targeted sibling of the broadcast ``/runtime/lifecycle``
    route: where that fans a notice across every bound session, this targets
    the single ``body.session_id`` — the SMS design's §3.5 req 1 (a delivery
    failure must reach the *originating* session, not be broadcast).

    Authorization mirrors ``post_runtime_tool_result`` exactly: the bearer's
    connector must match ``body.connection_id``'s connector, any bearer-side
    ``connection_ids`` allowlist must include it (#350), and the session must
    be genuinely bound to that connection — so a runtime bearer can only
    target a session within its own connections.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the exact pattern as the tool-result intake) so the failure wakes the
    session rather than merely being visible on its next turn.

    Reserved model-visible ``event`` values a connector may post here:
    ``connector_delivery_failed`` (#1308, the failure path), and its
    success-path complements ``connector_message_delivered`` /
    ``connector_message_edited`` (#1341, informational acks emitted with
    ``wake=False``). All three render as a bracketed user-role notice; any
    other ``event`` string is appended but filtered out of the model context
    by the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist.

    Args:
        authorization (None | str | Unset):
        body (RuntimeSessionLifecycleRequest): Body for ``POST /v1/connectors/runtime/session-
            lifecycle`` (#1261).

            The per-session-targeted sibling of :class:`RuntimeLifecycleRequest`.
            Where the broadcast ``/runtime/lifecycle`` route fans a transport-down
            notice across *every* session bound to the connection, this appends a
            single ``kind=lifecycle`` event onto **one** named session — the gap
            called out by the SMS design (§3.5 req 1): a delivery failure must reach
            the *originating* session, not be broadcast.

            ``wake`` optionally pairs the append with a ``defer_wake`` so the failure
            isn't merely visible-on-next-turn but actually wakes the session (the
            "give it stimulus" half of the design's option (a)). Defaults ``False``
            so the primitive stays a plain visible-on-next-wake append unless the
            caller opts into the wake.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle]
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
    body: RuntimeSessionLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
    | None
):
    """Post Runtime Session Lifecycle

     Append a ``kind=lifecycle`` event onto **one** session bound to
    ``body.connection_id`` (#1261), optionally waking it.

    The per-session-targeted sibling of the broadcast ``/runtime/lifecycle``
    route: where that fans a notice across every bound session, this targets
    the single ``body.session_id`` — the SMS design's §3.5 req 1 (a delivery
    failure must reach the *originating* session, not be broadcast).

    Authorization mirrors ``post_runtime_tool_result`` exactly: the bearer's
    connector must match ``body.connection_id``'s connector, any bearer-side
    ``connection_ids`` allowlist must include it (#350), and the session must
    be genuinely bound to that connection — so a runtime bearer can only
    target a session within its own connections.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the exact pattern as the tool-result intake) so the failure wakes the
    session rather than merely being visible on its next turn.

    Reserved model-visible ``event`` values a connector may post here:
    ``connector_delivery_failed`` (#1308, the failure path), and its
    success-path complements ``connector_message_delivered`` /
    ``connector_message_edited`` (#1341, informational acks emitted with
    ``wake=False``). All three render as a bracketed user-role notice; any
    other ``event`` string is appended but filtered out of the model context
    by the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist.

    Args:
        authorization (None | str | Unset):
        body (RuntimeSessionLifecycleRequest): Body for ``POST /v1/connectors/runtime/session-
            lifecycle`` (#1261).

            The per-session-targeted sibling of :class:`RuntimeLifecycleRequest`.
            Where the broadcast ``/runtime/lifecycle`` route fans a transport-down
            notice across *every* session bound to the connection, this appends a
            single ``kind=lifecycle`` event onto **one** named session — the gap
            called out by the SMS design (§3.5 req 1): a delivery failure must reach
            the *originating* session, not be broadcast.

            ``wake`` optionally pairs the append with a ``defer_wake`` so the failure
            isn't merely visible-on-next-turn but actually wakes the session (the
            "give it stimulus" half of the design's option (a)). Defaults ``False``
            so the primitive stays a plain visible-on-next-wake append unless the
            caller opts into the wake.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: RuntimeSessionLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
]:
    """Post Runtime Session Lifecycle

     Append a ``kind=lifecycle`` event onto **one** session bound to
    ``body.connection_id`` (#1261), optionally waking it.

    The per-session-targeted sibling of the broadcast ``/runtime/lifecycle``
    route: where that fans a notice across every bound session, this targets
    the single ``body.session_id`` — the SMS design's §3.5 req 1 (a delivery
    failure must reach the *originating* session, not be broadcast).

    Authorization mirrors ``post_runtime_tool_result`` exactly: the bearer's
    connector must match ``body.connection_id``'s connector, any bearer-side
    ``connection_ids`` allowlist must include it (#350), and the session must
    be genuinely bound to that connection — so a runtime bearer can only
    target a session within its own connections.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the exact pattern as the tool-result intake) so the failure wakes the
    session rather than merely being visible on its next turn.

    Reserved model-visible ``event`` values a connector may post here:
    ``connector_delivery_failed`` (#1308, the failure path), and its
    success-path complements ``connector_message_delivered`` /
    ``connector_message_edited`` (#1341, informational acks emitted with
    ``wake=False``). All three render as a bracketed user-role notice; any
    other ``event`` string is appended but filtered out of the model context
    by the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist.

    Args:
        authorization (None | str | Unset):
        body (RuntimeSessionLifecycleRequest): Body for ``POST /v1/connectors/runtime/session-
            lifecycle`` (#1261).

            The per-session-targeted sibling of :class:`RuntimeLifecycleRequest`.
            Where the broadcast ``/runtime/lifecycle`` route fans a transport-down
            notice across *every* session bound to the connection, this appends a
            single ``kind=lifecycle`` event onto **one** named session — the gap
            called out by the SMS design (§3.5 req 1): a delivery failure must reach
            the *originating* session, not be broadcast.

            ``wake`` optionally pairs the append with a ``defer_wake`` so the failure
            isn't merely visible-on-next-turn but actually wakes the session (the
            "give it stimulus" half of the design's option (a)). Defaults ``False``
            so the primitive stays a plain visible-on-next-wake append unless the
            caller opts into the wake.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle]
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
    body: RuntimeSessionLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
    | None
):
    """Post Runtime Session Lifecycle

     Append a ``kind=lifecycle`` event onto **one** session bound to
    ``body.connection_id`` (#1261), optionally waking it.

    The per-session-targeted sibling of the broadcast ``/runtime/lifecycle``
    route: where that fans a notice across every bound session, this targets
    the single ``body.session_id`` — the SMS design's §3.5 req 1 (a delivery
    failure must reach the *originating* session, not be broadcast).

    Authorization mirrors ``post_runtime_tool_result`` exactly: the bearer's
    connector must match ``body.connection_id``'s connector, any bearer-side
    ``connection_ids`` allowlist must include it (#350), and the session must
    be genuinely bound to that connection — so a runtime bearer can only
    target a session within its own connections.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the exact pattern as the tool-result intake) so the failure wakes the
    session rather than merely being visible on its next turn.

    Reserved model-visible ``event`` values a connector may post here:
    ``connector_delivery_failed`` (#1308, the failure path), and its
    success-path complements ``connector_message_delivered`` /
    ``connector_message_edited`` (#1341, informational acks emitted with
    ``wake=False``). All three render as a bracketed user-role notice; any
    other ``event`` string is appended but filtered out of the model context
    by the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist.

    Args:
        authorization (None | str | Unset):
        body (RuntimeSessionLifecycleRequest): Body for ``POST /v1/connectors/runtime/session-
            lifecycle`` (#1261).

            The per-session-targeted sibling of :class:`RuntimeLifecycleRequest`.
            Where the broadcast ``/runtime/lifecycle`` route fans a transport-down
            notice across *every* session bound to the connection, this appends a
            single ``kind=lifecycle`` event onto **one** named session — the gap
            called out by the SMS design (§3.5 req 1): a delivery failure must reach
            the *originating* session, not be broadcast.

            ``wake`` optionally pairs the append with a ``defer_wake`` so the failure
            isn't merely visible-on-next-turn but actually wakes the session (the
            "give it stimulus" half of the design's option (a)). Defaults ``False``
            so the primitive stays a plain visible-on-next-wake append unless the
            caller opts into the wake.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | PostConnectorRuntimeSessionLifecycleResponsePostConnectorRuntimeSessionLifecycle
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
