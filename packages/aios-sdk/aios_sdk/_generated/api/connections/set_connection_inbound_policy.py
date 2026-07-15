from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.allow_all import AllowAll
from ...models.allow_list import AllowList
from ...models.allow_senders import AllowSenders
from ...models.connection import Connection
from ...models.deny_all import DenyAll
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connection_id: str,
    *,
    body: AllowAll | AllowList | AllowSenders | DenyAll,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/connections/{connection_id}/inbound-policy".format(
            connection_id=quote(str(connection_id), safe=""),
        ),
    }

    if isinstance(body, AllowAll) or isinstance(body, AllowList) or isinstance(body, AllowSenders):
        _kwargs["json"] = body.to_dict()
    else:
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connection | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Connection.from_dict(response.json())

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
) -> Response[Connection | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AllowAll | AllowList | AllowSenders | DenyAll,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    r"""Set Inbound Policy

     Set the connection's inbound-admission policy, wholesale (Replace).

    Operator-authed (``AccountIdDep`` — **not** the runtime token; this
    route lives on the operator connections router, not the runtime-scoped
    connectors router). The body is the bare ``InboundPolicy`` union shape
    ``{\"kind\": ..., \"chat_ids\"?: [...]}`` with **Replace** semantics:

    * ``{\"kind\": \"allow_list\", \"chat_ids\": []}`` → **422** (empty list is
      never a silent deny-all; use ``deny_all``), never persisted.
    * ``{\"kind\": \"allow_list\"}`` (no ``chat_ids``) → **422** — ``chat_ids``
      is required-on-update, so a partial body can neither widen to
      allow-everyone nor silently re-default.
    * a body with no ``kind``, or a deferred/unknown ``kind`` (e.g.
      ``deny_list``) → **422** (discriminated-union + ``extra=\"forbid\"``).
    * ``{\"kind\": \"deny_all\"}`` / ``{\"kind\": \"allow_all\"}`` are accepted.

    Revocation is a Replace with the smaller ``AllowList``. Returns the
    updated ``Connection`` (its ``inbound_policy_effective`` reflects the
    new posture).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (AllowAll | AllowList | AllowSenders | DenyAll): Wire wrapper for ``PUT
            /v1/connections/{id}/inbound-policy``.

            A ``RootModel`` over the :data:`InboundPolicy` discriminated union so the
            request body is the bare ``{"kind": ..., "chat_ids"?: [...]}`` shape (no
            envelope key) and the validated member is reachable as ``body.root``.

            **Replace, not Patch.** This is the *required-on-update* variant:
            ``AllowList.chat_ids`` is required with ``min_length=1``, so a partial
            body ``{"kind": "allow_list"}`` (no ``chat_ids``) 422s rather than
            silently widening to an unbounded allow-everyone or re-defaulting, and an
            empty ``{"kind": "allow_list", "chat_ids": []}`` 422s at the write edge.
            ``DenyAll`` / ``AllowAll`` bodies carry no ``chat_ids`` and are accepted.
            An unknown or missing ``kind`` 422s via the discriminated-union
            validation plus each member's ``extra="forbid"``. (Mirrors
            ``TriggerSourceReplace`` in :mod:`aios.models.triggers`.)

            Revocation (§9) is expressed as a Replace with the smaller list — there
            is no separate patch shape.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AllowAll | AllowList | AllowSenders | DenyAll,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    r"""Set Inbound Policy

     Set the connection's inbound-admission policy, wholesale (Replace).

    Operator-authed (``AccountIdDep`` — **not** the runtime token; this
    route lives on the operator connections router, not the runtime-scoped
    connectors router). The body is the bare ``InboundPolicy`` union shape
    ``{\"kind\": ..., \"chat_ids\"?: [...]}`` with **Replace** semantics:

    * ``{\"kind\": \"allow_list\", \"chat_ids\": []}`` → **422** (empty list is
      never a silent deny-all; use ``deny_all``), never persisted.
    * ``{\"kind\": \"allow_list\"}`` (no ``chat_ids``) → **422** — ``chat_ids``
      is required-on-update, so a partial body can neither widen to
      allow-everyone nor silently re-default.
    * a body with no ``kind``, or a deferred/unknown ``kind`` (e.g.
      ``deny_list``) → **422** (discriminated-union + ``extra=\"forbid\"``).
    * ``{\"kind\": \"deny_all\"}`` / ``{\"kind\": \"allow_all\"}`` are accepted.

    Revocation is a Replace with the smaller ``AllowList``. Returns the
    updated ``Connection`` (its ``inbound_policy_effective`` reflects the
    new posture).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (AllowAll | AllowList | AllowSenders | DenyAll): Wire wrapper for ``PUT
            /v1/connections/{id}/inbound-policy``.

            A ``RootModel`` over the :data:`InboundPolicy` discriminated union so the
            request body is the bare ``{"kind": ..., "chat_ids"?: [...]}`` shape (no
            envelope key) and the validated member is reachable as ``body.root``.

            **Replace, not Patch.** This is the *required-on-update* variant:
            ``AllowList.chat_ids`` is required with ``min_length=1``, so a partial
            body ``{"kind": "allow_list"}`` (no ``chat_ids``) 422s rather than
            silently widening to an unbounded allow-everyone or re-defaulting, and an
            empty ``{"kind": "allow_list", "chat_ids": []}`` 422s at the write edge.
            ``DenyAll`` / ``AllowAll`` bodies carry no ``chat_ids`` and are accepted.
            An unknown or missing ``kind`` 422s via the discriminated-union
            validation plus each member's ``extra="forbid"``. (Mirrors
            ``TriggerSourceReplace`` in :mod:`aios.models.triggers`.)

            Revocation (§9) is expressed as a Replace with the smaller list — there
            is no separate patch shape.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return sync_detailed(
        connection_id=connection_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AllowAll | AllowList | AllowSenders | DenyAll,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    r"""Set Inbound Policy

     Set the connection's inbound-admission policy, wholesale (Replace).

    Operator-authed (``AccountIdDep`` — **not** the runtime token; this
    route lives on the operator connections router, not the runtime-scoped
    connectors router). The body is the bare ``InboundPolicy`` union shape
    ``{\"kind\": ..., \"chat_ids\"?: [...]}`` with **Replace** semantics:

    * ``{\"kind\": \"allow_list\", \"chat_ids\": []}`` → **422** (empty list is
      never a silent deny-all; use ``deny_all``), never persisted.
    * ``{\"kind\": \"allow_list\"}`` (no ``chat_ids``) → **422** — ``chat_ids``
      is required-on-update, so a partial body can neither widen to
      allow-everyone nor silently re-default.
    * a body with no ``kind``, or a deferred/unknown ``kind`` (e.g.
      ``deny_list``) → **422** (discriminated-union + ``extra=\"forbid\"``).
    * ``{\"kind\": \"deny_all\"}`` / ``{\"kind\": \"allow_all\"}`` are accepted.

    Revocation is a Replace with the smaller ``AllowList``. Returns the
    updated ``Connection`` (its ``inbound_policy_effective`` reflects the
    new posture).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (AllowAll | AllowList | AllowSenders | DenyAll): Wire wrapper for ``PUT
            /v1/connections/{id}/inbound-policy``.

            A ``RootModel`` over the :data:`InboundPolicy` discriminated union so the
            request body is the bare ``{"kind": ..., "chat_ids"?: [...]}`` shape (no
            envelope key) and the validated member is reachable as ``body.root``.

            **Replace, not Patch.** This is the *required-on-update* variant:
            ``AllowList.chat_ids`` is required with ``min_length=1``, so a partial
            body ``{"kind": "allow_list"}`` (no ``chat_ids``) 422s rather than
            silently widening to an unbounded allow-everyone or re-defaulting, and an
            empty ``{"kind": "allow_list", "chat_ids": []}`` 422s at the write edge.
            ``DenyAll`` / ``AllowAll`` bodies carry no ``chat_ids`` and are accepted.
            An unknown or missing ``kind`` 422s via the discriminated-union
            validation plus each member's ``extra="forbid"``. (Mirrors
            ``TriggerSourceReplace`` in :mod:`aios.models.triggers`.)

            Revocation (§9) is expressed as a Replace with the smaller list — there
            is no separate patch shape.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AllowAll | AllowList | AllowSenders | DenyAll,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    r"""Set Inbound Policy

     Set the connection's inbound-admission policy, wholesale (Replace).

    Operator-authed (``AccountIdDep`` — **not** the runtime token; this
    route lives on the operator connections router, not the runtime-scoped
    connectors router). The body is the bare ``InboundPolicy`` union shape
    ``{\"kind\": ..., \"chat_ids\"?: [...]}`` with **Replace** semantics:

    * ``{\"kind\": \"allow_list\", \"chat_ids\": []}`` → **422** (empty list is
      never a silent deny-all; use ``deny_all``), never persisted.
    * ``{\"kind\": \"allow_list\"}`` (no ``chat_ids``) → **422** — ``chat_ids``
      is required-on-update, so a partial body can neither widen to
      allow-everyone nor silently re-default.
    * a body with no ``kind``, or a deferred/unknown ``kind`` (e.g.
      ``deny_list``) → **422** (discriminated-union + ``extra=\"forbid\"``).
    * ``{\"kind\": \"deny_all\"}`` / ``{\"kind\": \"allow_all\"}`` are accepted.

    Revocation is a Replace with the smaller ``AllowList``. Returns the
    updated ``Connection`` (its ``inbound_policy_effective`` reflects the
    new posture).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (AllowAll | AllowList | AllowSenders | DenyAll): Wire wrapper for ``PUT
            /v1/connections/{id}/inbound-policy``.

            A ``RootModel`` over the :data:`InboundPolicy` discriminated union so the
            request body is the bare ``{"kind": ..., "chat_ids"?: [...]}`` shape (no
            envelope key) and the validated member is reachable as ``body.root``.

            **Replace, not Patch.** This is the *required-on-update* variant:
            ``AllowList.chat_ids`` is required with ``min_length=1``, so a partial
            body ``{"kind": "allow_list"}`` (no ``chat_ids``) 422s rather than
            silently widening to an unbounded allow-everyone or re-defaulting, and an
            empty ``{"kind": "allow_list", "chat_ids": []}`` 422s at the write edge.
            ``DenyAll`` / ``AllowAll`` bodies carry no ``chat_ids`` and are accepted.
            An unknown or missing ``kind`` 422s via the discriminated-union
            validation plus each member's ``extra="forbid"``. (Mirrors
            ``TriggerSourceReplace`` in :mod:`aios.models.triggers`.)

            Revocation (§9) is expressed as a Replace with the smaller list — there
            is no separate patch shape.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            connection_id=connection_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
