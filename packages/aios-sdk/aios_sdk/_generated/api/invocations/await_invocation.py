from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.await_response import AwaitResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    task_id: str,
    *,
    request_id: None | str | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_request_id: None | str | Unset
    if isinstance(request_id, Unset):
        json_request_id = UNSET
    else:
        json_request_id = request_id
    params["request_id"] = json_request_id

    params["timeout"] = timeout

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/invocations/{task_id}/await".format(
            task_id=quote(str(task_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AwaitResponse | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = AwaitResponse.from_dict(response.json())

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
) -> Response[AwaitResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: None | str | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[AwaitResponse | HTTPValidationError]:
    """Await Invocation

     Block until the invocation reaches a terminal state, or ``timeout`` seconds.

    The **one awaiter** over both servicer kinds: ``task_id`` is the ``servicer_id``
    from the POST handle and its kind is read off the id prefix. A ``session``
    servicer needs ``?request_id=`` to correlate its response; a ``run`` resolves off
    its terminal row (``request_id`` ignored). On timeout returns ``outcome=null`` so
    the caller re-polls — a plain request/response (MCP-usable) so an agent can await
    a sub-invocation and join. A cross-tenant/missing servicer 404s.

    Args:
        task_id (str):
        request_id (None | str | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AwaitResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        task_id=task_id,
        request_id=request_id,
        timeout=timeout,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: None | str | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> AwaitResponse | HTTPValidationError | None:
    """Await Invocation

     Block until the invocation reaches a terminal state, or ``timeout`` seconds.

    The **one awaiter** over both servicer kinds: ``task_id`` is the ``servicer_id``
    from the POST handle and its kind is read off the id prefix. A ``session``
    servicer needs ``?request_id=`` to correlate its response; a ``run`` resolves off
    its terminal row (``request_id`` ignored). On timeout returns ``outcome=null`` so
    the caller re-polls — a plain request/response (MCP-usable) so an agent can await
    a sub-invocation and join. A cross-tenant/missing servicer 404s.

    Args:
        task_id (str):
        request_id (None | str | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AwaitResponse | HTTPValidationError
    """

    return sync_detailed(
        task_id=task_id,
        client=client,
        request_id=request_id,
        timeout=timeout,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: None | str | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[AwaitResponse | HTTPValidationError]:
    """Await Invocation

     Block until the invocation reaches a terminal state, or ``timeout`` seconds.

    The **one awaiter** over both servicer kinds: ``task_id`` is the ``servicer_id``
    from the POST handle and its kind is read off the id prefix. A ``session``
    servicer needs ``?request_id=`` to correlate its response; a ``run`` resolves off
    its terminal row (``request_id`` ignored). On timeout returns ``outcome=null`` so
    the caller re-polls — a plain request/response (MCP-usable) so an agent can await
    a sub-invocation and join. A cross-tenant/missing servicer 404s.

    Args:
        task_id (str):
        request_id (None | str | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AwaitResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        task_id=task_id,
        request_id=request_id,
        timeout=timeout,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: None | str | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> AwaitResponse | HTTPValidationError | None:
    """Await Invocation

     Block until the invocation reaches a terminal state, or ``timeout`` seconds.

    The **one awaiter** over both servicer kinds: ``task_id`` is the ``servicer_id``
    from the POST handle and its kind is read off the id prefix. A ``session``
    servicer needs ``?request_id=`` to correlate its response; a ``run`` resolves off
    its terminal row (``request_id`` ignored). On timeout returns ``outcome=null`` so
    the caller re-polls — a plain request/response (MCP-usable) so an agent can await
    a sub-invocation and join. A cross-tenant/missing servicer 404s.

    Args:
        task_id (str):
        request_id (None | str | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AwaitResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            task_id=task_id,
            client=client,
            request_id=request_id,
            timeout=timeout,
            authorization=authorization,
        )
    ).parsed
