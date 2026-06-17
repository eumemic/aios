from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.post_connector_runtime_chat_lifecycle_response_post_connector_runtime_chat_lifecycle import (
    PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle,
)
from ...models.runtime_chat_lifecycle_request import RuntimeChatLifecycleRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: RuntimeChatLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/runtime/chat-lifecycle",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    HTTPValidationError
    | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
    | None
):
    if response.status_code == 201:
        response_201 = PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle.from_dict(
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
    | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
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
    body: RuntimeChatLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
]:
    r"""Post Runtime Chat Lifecycle

     Append a ``kind=lifecycle`` event onto the single session that
    ``body.chat_id`` resolves to on ``body.connection_id`` (#1260),
    optionally waking it.

    The routing-key sibling of ``/runtime/session-lifecycle``: where that
    needs the resolved ``session_id``, this carries the connector's per-peer
    routing key (``chat_id``) and resolves it through the connection's
    per-chat binding server-side — the SMS design's §3.5 req 1 second option
    (\"route the per-peer failure through the resolver on the callback's
    ``To``\").  Like the session-lifecycle route it targets exactly one
    session, NOT the broadcast fan-out: a per-peer delivery failure must not
    pollute unrelated ``per_chat`` sessions.

    Authorization mirrors the session-lifecycle route: the bearer's
    connector must match ``body.connection_id``'s connector and any
    bearer-side ``connection_ids`` allowlist must include it (#350).  The
    binding lookup itself is the per-session authorization — a ``chat_id``
    that has no per-chat session on this connection 404s (no spurious
    cross-peer append), and the resolution is scoped to the bearer's
    ``account_id``.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the same pattern as the session-lifecycle and tool-result intakes) so
    the failure wakes the originating session rather than merely being
    visible on its next turn.

    Args:
        authorization (None | str | Unset):
        body (RuntimeChatLifecycleRequest): Body for ``POST /v1/connectors/runtime/chat-
            lifecycle`` (#1260).

            The routing-key variant of :class:`RuntimeSessionLifecycleRequest`.
            Both target a *single* session (not the broadcast fan-out), but where
            the session-lifecycle route needs the caller to already hold the
            resolved ``session_id``, this route carries a per-peer **routing key**
            (``chat_id``) and resolves it through the connection's per-chat binding
            to the originating session server-side.

            This is the second option the SMS design (§3.5 req 1) calls out: "route
            the per-peer failure through the resolver on the callback's ``To``".  A
            Twilio status callback knows the peer number (→ ``chat_id``) but not the
            AIOS ``session_id`` — without this route the connector would have to do
            an extra round-trip (or maintain its own ``chat_id → session_id`` map)
            just to reach the originating per_chat session.  The broadcast
            ``/runtime/lifecycle`` route stays for genuine connection-wide events.

            ``chat_id`` is the connector's per-peer routing key, the same value the
            inbound path stamps onto ``chat_sessions``.  It must resolve to an
            existing per-chat binding on ``connection_id`` — a routing key with no
            bound session 404s rather than fanning a spurious cross-peer notice (the
            design's "if a correlation row is genuinely missing … drop rather than
            fan a spurious cross-peer failure", §3.5).

            ``wake`` mirrors the session-lifecycle route: ``True`` pairs the append
            with a ``defer_wake`` so the failure wakes the originating session;
            defaults ``False`` (visible-on-next-turn).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle]
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
    body: RuntimeChatLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
    | None
):
    r"""Post Runtime Chat Lifecycle

     Append a ``kind=lifecycle`` event onto the single session that
    ``body.chat_id`` resolves to on ``body.connection_id`` (#1260),
    optionally waking it.

    The routing-key sibling of ``/runtime/session-lifecycle``: where that
    needs the resolved ``session_id``, this carries the connector's per-peer
    routing key (``chat_id``) and resolves it through the connection's
    per-chat binding server-side — the SMS design's §3.5 req 1 second option
    (\"route the per-peer failure through the resolver on the callback's
    ``To``\").  Like the session-lifecycle route it targets exactly one
    session, NOT the broadcast fan-out: a per-peer delivery failure must not
    pollute unrelated ``per_chat`` sessions.

    Authorization mirrors the session-lifecycle route: the bearer's
    connector must match ``body.connection_id``'s connector and any
    bearer-side ``connection_ids`` allowlist must include it (#350).  The
    binding lookup itself is the per-session authorization — a ``chat_id``
    that has no per-chat session on this connection 404s (no spurious
    cross-peer append), and the resolution is scoped to the bearer's
    ``account_id``.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the same pattern as the session-lifecycle and tool-result intakes) so
    the failure wakes the originating session rather than merely being
    visible on its next turn.

    Args:
        authorization (None | str | Unset):
        body (RuntimeChatLifecycleRequest): Body for ``POST /v1/connectors/runtime/chat-
            lifecycle`` (#1260).

            The routing-key variant of :class:`RuntimeSessionLifecycleRequest`.
            Both target a *single* session (not the broadcast fan-out), but where
            the session-lifecycle route needs the caller to already hold the
            resolved ``session_id``, this route carries a per-peer **routing key**
            (``chat_id``) and resolves it through the connection's per-chat binding
            to the originating session server-side.

            This is the second option the SMS design (§3.5 req 1) calls out: "route
            the per-peer failure through the resolver on the callback's ``To``".  A
            Twilio status callback knows the peer number (→ ``chat_id``) but not the
            AIOS ``session_id`` — without this route the connector would have to do
            an extra round-trip (or maintain its own ``chat_id → session_id`` map)
            just to reach the originating per_chat session.  The broadcast
            ``/runtime/lifecycle`` route stays for genuine connection-wide events.

            ``chat_id`` is the connector's per-peer routing key, the same value the
            inbound path stamps onto ``chat_sessions``.  It must resolve to an
            existing per-chat binding on ``connection_id`` — a routing key with no
            bound session 404s rather than fanning a spurious cross-peer notice (the
            design's "if a correlation row is genuinely missing … drop rather than
            fan a spurious cross-peer failure", §3.5).

            ``wake`` mirrors the session-lifecycle route: ``True`` pairs the append
            with a ``defer_wake`` so the failure wakes the originating session;
            defaults ``False`` (visible-on-next-turn).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: RuntimeChatLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
]:
    r"""Post Runtime Chat Lifecycle

     Append a ``kind=lifecycle`` event onto the single session that
    ``body.chat_id`` resolves to on ``body.connection_id`` (#1260),
    optionally waking it.

    The routing-key sibling of ``/runtime/session-lifecycle``: where that
    needs the resolved ``session_id``, this carries the connector's per-peer
    routing key (``chat_id``) and resolves it through the connection's
    per-chat binding server-side — the SMS design's §3.5 req 1 second option
    (\"route the per-peer failure through the resolver on the callback's
    ``To``\").  Like the session-lifecycle route it targets exactly one
    session, NOT the broadcast fan-out: a per-peer delivery failure must not
    pollute unrelated ``per_chat`` sessions.

    Authorization mirrors the session-lifecycle route: the bearer's
    connector must match ``body.connection_id``'s connector and any
    bearer-side ``connection_ids`` allowlist must include it (#350).  The
    binding lookup itself is the per-session authorization — a ``chat_id``
    that has no per-chat session on this connection 404s (no spurious
    cross-peer append), and the resolution is scoped to the bearer's
    ``account_id``.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the same pattern as the session-lifecycle and tool-result intakes) so
    the failure wakes the originating session rather than merely being
    visible on its next turn.

    Args:
        authorization (None | str | Unset):
        body (RuntimeChatLifecycleRequest): Body for ``POST /v1/connectors/runtime/chat-
            lifecycle`` (#1260).

            The routing-key variant of :class:`RuntimeSessionLifecycleRequest`.
            Both target a *single* session (not the broadcast fan-out), but where
            the session-lifecycle route needs the caller to already hold the
            resolved ``session_id``, this route carries a per-peer **routing key**
            (``chat_id``) and resolves it through the connection's per-chat binding
            to the originating session server-side.

            This is the second option the SMS design (§3.5 req 1) calls out: "route
            the per-peer failure through the resolver on the callback's ``To``".  A
            Twilio status callback knows the peer number (→ ``chat_id``) but not the
            AIOS ``session_id`` — without this route the connector would have to do
            an extra round-trip (or maintain its own ``chat_id → session_id`` map)
            just to reach the originating per_chat session.  The broadcast
            ``/runtime/lifecycle`` route stays for genuine connection-wide events.

            ``chat_id`` is the connector's per-peer routing key, the same value the
            inbound path stamps onto ``chat_sessions``.  It must resolve to an
            existing per-chat binding on ``connection_id`` — a routing key with no
            bound session 404s rather than fanning a spurious cross-peer notice (the
            design's "if a correlation row is genuinely missing … drop rather than
            fan a spurious cross-peer failure", §3.5).

            ``wake`` mirrors the session-lifecycle route: ``True`` pairs the append
            with a ``defer_wake`` so the failure wakes the originating session;
            defaults ``False`` (visible-on-next-turn).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle]
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
    body: RuntimeChatLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
    | None
):
    r"""Post Runtime Chat Lifecycle

     Append a ``kind=lifecycle`` event onto the single session that
    ``body.chat_id`` resolves to on ``body.connection_id`` (#1260),
    optionally waking it.

    The routing-key sibling of ``/runtime/session-lifecycle``: where that
    needs the resolved ``session_id``, this carries the connector's per-peer
    routing key (``chat_id``) and resolves it through the connection's
    per-chat binding server-side — the SMS design's §3.5 req 1 second option
    (\"route the per-peer failure through the resolver on the callback's
    ``To``\").  Like the session-lifecycle route it targets exactly one
    session, NOT the broadcast fan-out: a per-peer delivery failure must not
    pollute unrelated ``per_chat`` sessions.

    Authorization mirrors the session-lifecycle route: the bearer's
    connector must match ``body.connection_id``'s connector and any
    bearer-side ``connection_ids`` allowlist must include it (#350).  The
    binding lookup itself is the per-session authorization — a ``chat_id``
    that has no per-chat session on this connection 404s (no spurious
    cross-peer append), and the resolution is scoped to the bearer's
    ``account_id``.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the same pattern as the session-lifecycle and tool-result intakes) so
    the failure wakes the originating session rather than merely being
    visible on its next turn.

    Args:
        authorization (None | str | Unset):
        body (RuntimeChatLifecycleRequest): Body for ``POST /v1/connectors/runtime/chat-
            lifecycle`` (#1260).

            The routing-key variant of :class:`RuntimeSessionLifecycleRequest`.
            Both target a *single* session (not the broadcast fan-out), but where
            the session-lifecycle route needs the caller to already hold the
            resolved ``session_id``, this route carries a per-peer **routing key**
            (``chat_id``) and resolves it through the connection's per-chat binding
            to the originating session server-side.

            This is the second option the SMS design (§3.5 req 1) calls out: "route
            the per-peer failure through the resolver on the callback's ``To``".  A
            Twilio status callback knows the peer number (→ ``chat_id``) but not the
            AIOS ``session_id`` — without this route the connector would have to do
            an extra round-trip (or maintain its own ``chat_id → session_id`` map)
            just to reach the originating per_chat session.  The broadcast
            ``/runtime/lifecycle`` route stays for genuine connection-wide events.

            ``chat_id`` is the connector's per-peer routing key, the same value the
            inbound path stamps onto ``chat_sessions``.  It must resolve to an
            existing per-chat binding on ``connection_id`` — a routing key with no
            bound session 404s rather than fanning a spurious cross-peer notice (the
            design's "if a correlation row is genuinely missing … drop rather than
            fan a spurious cross-peer failure", §3.5).

            ``wake`` mirrors the session-lifecycle route: ``True`` pairs the append
            with a ``defer_wake`` so the failure wakes the originating session;
            defaults ``False`` (visible-on-next-turn).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | PostConnectorRuntimeChatLifecycleResponsePostConnectorRuntimeChatLifecycle
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
