from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.post_connector_runtime_lifecycle_response_post_connector_runtime_lifecycle import (
    PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle,
)
from ...models.runtime_lifecycle_request import RuntimeLifecycleRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: RuntimeLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/runtime/lifecycle",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    HTTPValidationError
    | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
    | None
):
    if response.status_code == 201:
        response_201 = PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle.from_dict(
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
    | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
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
    body: RuntimeLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
]:
    r"""Post Runtime Lifecycle

     Append a ``kind=lifecycle`` event onto every session bound to
    ``body.connection_id``.

    Authorization mirrors the inbound + tool-result paths: the bearer's
    connector must match ``body.connection_id``'s connector, and any
    bearer-side allowlist must include the connection_id (#350).

    Returns ``{\"appended_session_ids\": [...]}`` enumerating the sessions
    that received the event.  An empty list means no sessions were
    bound at the time of the call (e.g. the operator detached every
    session before the connector finished tearing down); not an error.

    Args:
        authorization (None | str | Unset):
        body (RuntimeLifecycleRequest): Body for ``POST /v1/connectors/runtime/lifecycle``.

            Lets a connector emit a lifecycle event onto each session bound to
            ``connection_id`` — used today for "the underlying transport just
            went away" notifications (WhatsApp daemon crashed, peer logged the
            device out, etc.) so the model sees the connection-broken state in
            its context instead of silently failing the next outbound.

            ``event`` is a connector-namespaced kind ("whatsapp.connection.lost",
            "signal.daemon.exited") — the connector chooses the vocabulary.
            ``reason`` is an optional short tag the harness surfaces alongside
            the event for the model to act on ("daemon_crashed", "peer_logout").
            ``data`` is an optional free-form dict for connector-specific
            context (current device count, last successful timestamp, etc.).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle]
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
    body: RuntimeLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
    | None
):
    r"""Post Runtime Lifecycle

     Append a ``kind=lifecycle`` event onto every session bound to
    ``body.connection_id``.

    Authorization mirrors the inbound + tool-result paths: the bearer's
    connector must match ``body.connection_id``'s connector, and any
    bearer-side allowlist must include the connection_id (#350).

    Returns ``{\"appended_session_ids\": [...]}`` enumerating the sessions
    that received the event.  An empty list means no sessions were
    bound at the time of the call (e.g. the operator detached every
    session before the connector finished tearing down); not an error.

    Args:
        authorization (None | str | Unset):
        body (RuntimeLifecycleRequest): Body for ``POST /v1/connectors/runtime/lifecycle``.

            Lets a connector emit a lifecycle event onto each session bound to
            ``connection_id`` — used today for "the underlying transport just
            went away" notifications (WhatsApp daemon crashed, peer logged the
            device out, etc.) so the model sees the connection-broken state in
            its context instead of silently failing the next outbound.

            ``event`` is a connector-namespaced kind ("whatsapp.connection.lost",
            "signal.daemon.exited") — the connector chooses the vocabulary.
            ``reason`` is an optional short tag the harness surfaces alongside
            the event for the model to act on ("daemon_crashed", "peer_logout").
            ``data`` is an optional free-form dict for connector-specific
            context (current device count, last successful timestamp, etc.).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: RuntimeLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
]:
    r"""Post Runtime Lifecycle

     Append a ``kind=lifecycle`` event onto every session bound to
    ``body.connection_id``.

    Authorization mirrors the inbound + tool-result paths: the bearer's
    connector must match ``body.connection_id``'s connector, and any
    bearer-side allowlist must include the connection_id (#350).

    Returns ``{\"appended_session_ids\": [...]}`` enumerating the sessions
    that received the event.  An empty list means no sessions were
    bound at the time of the call (e.g. the operator detached every
    session before the connector finished tearing down); not an error.

    Args:
        authorization (None | str | Unset):
        body (RuntimeLifecycleRequest): Body for ``POST /v1/connectors/runtime/lifecycle``.

            Lets a connector emit a lifecycle event onto each session bound to
            ``connection_id`` — used today for "the underlying transport just
            went away" notifications (WhatsApp daemon crashed, peer logged the
            device out, etc.) so the model sees the connection-broken state in
            its context instead of silently failing the next outbound.

            ``event`` is a connector-namespaced kind ("whatsapp.connection.lost",
            "signal.daemon.exited") — the connector chooses the vocabulary.
            ``reason`` is an optional short tag the harness surfaces alongside
            the event for the model to act on ("daemon_crashed", "peer_logout").
            ``data`` is an optional free-form dict for connector-specific
            context (current device count, last successful timestamp, etc.).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle]
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
    body: RuntimeLifecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
    | None
):
    r"""Post Runtime Lifecycle

     Append a ``kind=lifecycle`` event onto every session bound to
    ``body.connection_id``.

    Authorization mirrors the inbound + tool-result paths: the bearer's
    connector must match ``body.connection_id``'s connector, and any
    bearer-side allowlist must include the connection_id (#350).

    Returns ``{\"appended_session_ids\": [...]}`` enumerating the sessions
    that received the event.  An empty list means no sessions were
    bound at the time of the call (e.g. the operator detached every
    session before the connector finished tearing down); not an error.

    Args:
        authorization (None | str | Unset):
        body (RuntimeLifecycleRequest): Body for ``POST /v1/connectors/runtime/lifecycle``.

            Lets a connector emit a lifecycle event onto each session bound to
            ``connection_id`` — used today for "the underlying transport just
            went away" notifications (WhatsApp daemon crashed, peer logged the
            device out, etc.) so the model sees the connection-broken state in
            its context instead of silently failing the next outbound.

            ``event`` is a connector-namespaced kind ("whatsapp.connection.lost",
            "signal.daemon.exited") — the connector chooses the vocabulary.
            ``reason`` is an optional short tag the harness surfaces alongside
            the event for the model to act on ("daemon_crashed", "peer_logout").
            ``data`` is an optional free-form dict for connector-specific
            context (current device count, last successful timestamp, etc.).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | PostConnectorRuntimeLifecycleResponsePostConnectorRuntimeLifecycle
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
