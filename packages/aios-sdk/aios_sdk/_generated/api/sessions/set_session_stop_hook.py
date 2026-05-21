from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.session import Session
from ...models.stop_hook_request import StopHookRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: StopHookRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/stop-hook".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

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
    *,
    client: AuthenticatedClient | Client,
    body: StopHookRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Session]:
    r"""Set Stop Hook

     Set or clear the session's pluggable stop hook.

    Pass ``{\"hook\": null}`` (or omit ``hook``) to clear and return the
    session to its conversational default.  The harness's next
    conversational stop-point picks up the new hook.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (StopHookRequest): Request body for ``POST /v1/sessions/{id}/stop-hook``.

            ``hook=None`` clears the hook (returns the session to conversational
            default).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Session]
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
    body: StopHookRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Session | None:
    r"""Set Stop Hook

     Set or clear the session's pluggable stop hook.

    Pass ``{\"hook\": null}`` (or omit ``hook``) to clear and return the
    session to its conversational default.  The harness's next
    conversational stop-point picks up the new hook.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (StopHookRequest): Request body for ``POST /v1/sessions/{id}/stop-hook``.

            ``hook=None`` clears the hook (returns the session to conversational
            default).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Session
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
    body: StopHookRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Session]:
    r"""Set Stop Hook

     Set or clear the session's pluggable stop hook.

    Pass ``{\"hook\": null}`` (or omit ``hook``) to clear and return the
    session to its conversational default.  The harness's next
    conversational stop-point picks up the new hook.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (StopHookRequest): Request body for ``POST /v1/sessions/{id}/stop-hook``.

            ``hook=None`` clears the hook (returns the session to conversational
            default).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Session]
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
    body: StopHookRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Session | None:
    r"""Set Stop Hook

     Set or clear the session's pluggable stop hook.

    Pass ``{\"hook\": null}`` (or omit ``hook``) to clear and return the
    session to its conversational default.  The harness's next
    conversational stop-point picks up the new hook.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (StopHookRequest): Request body for ``POST /v1/sessions/{id}/stop-hook``.

            ``hook=None`` clears the hook (returns the session to conversational
            default).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Session
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
