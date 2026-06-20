from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.session import Session
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    goal_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/goals/{goal_id}/cancel".format(
            session_id=quote(str(session_id), safe=""),
            goal_id=quote(str(goal_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Session | None:
    if response.status_code == 200:
        response_200 = Session.from_dict(response.json())

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
) -> Response[HTTPValidationError | Session]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    goal_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Session]:
    r"""Cancel Goal

     Operator-cancel a session's own standing goal (#1414).

    Mirrors ``interrupt`` but targets a single self-issued goal rather than all
    in-flight work. **SECURITY (mandatory):** account-scope the session FIRST via
    ``service.get_session`` (404s a cross-tenant ``session_id``) — because the
    underlying ``respond_to_request`` resolves ``account_id`` UNSCOPED off the
    target row, so a cross-tenant id would otherwise write into another tenant's
    session. Then verify ``goal_id`` is a genuine self-goal (``caller=={kind:
    session, id:session_id}``) and write ``error={kind:\"cancelled\",
    by:\"operator\"}``; a non-self obligation (peer-invoke / workflow-child) 404s
    rather than being stamped. **Goals-only for v1** — the generic
    ``/requests/{id}/cancel`` is deferred (#1152 cancel-cascade). Enumerate a
    session's open ``goal_id``s via the ``owed_requests`` read-model on
    ``GET /v1/sessions/{id}``.

    Args:
        session_id (str):
        goal_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Session]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        goal_id=goal_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    goal_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Session | None:
    r"""Cancel Goal

     Operator-cancel a session's own standing goal (#1414).

    Mirrors ``interrupt`` but targets a single self-issued goal rather than all
    in-flight work. **SECURITY (mandatory):** account-scope the session FIRST via
    ``service.get_session`` (404s a cross-tenant ``session_id``) — because the
    underlying ``respond_to_request`` resolves ``account_id`` UNSCOPED off the
    target row, so a cross-tenant id would otherwise write into another tenant's
    session. Then verify ``goal_id`` is a genuine self-goal (``caller=={kind:
    session, id:session_id}``) and write ``error={kind:\"cancelled\",
    by:\"operator\"}``; a non-self obligation (peer-invoke / workflow-child) 404s
    rather than being stamped. **Goals-only for v1** — the generic
    ``/requests/{id}/cancel`` is deferred (#1152 cancel-cascade). Enumerate a
    session's open ``goal_id``s via the ``owed_requests`` read-model on
    ``GET /v1/sessions/{id}``.

    Args:
        session_id (str):
        goal_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Session
    """

    return sync_detailed(
        session_id=session_id,
        goal_id=goal_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    goal_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Session]:
    r"""Cancel Goal

     Operator-cancel a session's own standing goal (#1414).

    Mirrors ``interrupt`` but targets a single self-issued goal rather than all
    in-flight work. **SECURITY (mandatory):** account-scope the session FIRST via
    ``service.get_session`` (404s a cross-tenant ``session_id``) — because the
    underlying ``respond_to_request`` resolves ``account_id`` UNSCOPED off the
    target row, so a cross-tenant id would otherwise write into another tenant's
    session. Then verify ``goal_id`` is a genuine self-goal (``caller=={kind:
    session, id:session_id}``) and write ``error={kind:\"cancelled\",
    by:\"operator\"}``; a non-self obligation (peer-invoke / workflow-child) 404s
    rather than being stamped. **Goals-only for v1** — the generic
    ``/requests/{id}/cancel`` is deferred (#1152 cancel-cascade). Enumerate a
    session's open ``goal_id``s via the ``owed_requests`` read-model on
    ``GET /v1/sessions/{id}``.

    Args:
        session_id (str):
        goal_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Session]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        goal_id=goal_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    goal_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Session | None:
    r"""Cancel Goal

     Operator-cancel a session's own standing goal (#1414).

    Mirrors ``interrupt`` but targets a single self-issued goal rather than all
    in-flight work. **SECURITY (mandatory):** account-scope the session FIRST via
    ``service.get_session`` (404s a cross-tenant ``session_id``) — because the
    underlying ``respond_to_request`` resolves ``account_id`` UNSCOPED off the
    target row, so a cross-tenant id would otherwise write into another tenant's
    session. Then verify ``goal_id`` is a genuine self-goal (``caller=={kind:
    session, id:session_id}``) and write ``error={kind:\"cancelled\",
    by:\"operator\"}``; a non-self obligation (peer-invoke / workflow-child) 404s
    rather than being stamped. **Goals-only for v1** — the generic
    ``/requests/{id}/cancel`` is deferred (#1152 cancel-cascade). Enumerate a
    session's open ``goal_id``s via the ``owed_requests`` read-model on
    ``GET /v1/sessions/{id}``.

    Args:
        session_id (str):
        goal_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Session
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            goal_id=goal_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
