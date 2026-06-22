from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    task_id: str,
    *,
    request_id: str,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["request_id"] = request_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/tasks/{task_id}/cancel".format(
            task_id=quote(str(task_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 202:
        response_202 = response.json()
        return response_202

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
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: str,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Cancel Task

     Cancel the task identified by ``task_id`` (the servicer id) + ``request_id``.

    Seeds the cancel on the servicer (a run via its cancel signal, a session via its
    cancel-marker); the servicer harvests it under its own single-writer step lock and answers
    the request ``cancelled``. **202 Accepted**: the cancel is *requested*, finalized on the
    servicer's next step. Account-scoped; idempotent (re-cancel is a no-op); a cross-tenant or
    missing servicer 404s. ``task_id``'s kind is read off the id prefix (``wfr``/``sess``).

    Args:
        task_id (str):
        request_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        task_id=task_id,
        request_id=request_id,
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
    request_id: str,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Cancel Task

     Cancel the task identified by ``task_id`` (the servicer id) + ``request_id``.

    Seeds the cancel on the servicer (a run via its cancel signal, a session via its
    cancel-marker); the servicer harvests it under its own single-writer step lock and answers
    the request ``cancelled``. **202 Accepted**: the cancel is *requested*, finalized on the
    servicer's next step. Account-scoped; idempotent (re-cancel is a no-op); a cross-tenant or
    missing servicer 404s. ``task_id``'s kind is read off the id prefix (``wfr``/``sess``).

    Args:
        task_id (str):
        request_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        task_id=task_id,
        client=client,
        request_id=request_id,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: str,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Cancel Task

     Cancel the task identified by ``task_id`` (the servicer id) + ``request_id``.

    Seeds the cancel on the servicer (a run via its cancel signal, a session via its
    cancel-marker); the servicer harvests it under its own single-writer step lock and answers
    the request ``cancelled``. **202 Accepted**: the cancel is *requested*, finalized on the
    servicer's next step. Account-scoped; idempotent (re-cancel is a no-op); a cross-tenant or
    missing servicer 404s. ``task_id``'s kind is read off the id prefix (``wfr``/``sess``).

    Args:
        task_id (str):
        request_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        task_id=task_id,
        request_id=request_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    task_id: str,
    *,
    client: AuthenticatedClient | Client,
    request_id: str,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Cancel Task

     Cancel the task identified by ``task_id`` (the servicer id) + ``request_id``.

    Seeds the cancel on the servicer (a run via its cancel signal, a session via its
    cancel-marker); the servicer harvests it under its own single-writer step lock and answers
    the request ``cancelled``. **202 Accepted**: the cancel is *requested*, finalized on the
    servicer's next step. Account-scoped; idempotent (re-cancel is a no-op); a cross-tenant or
    missing servicer 404s. ``task_id``'s kind is read off the id prefix (``wfr``/``sess``).

    Args:
        task_id (str):
        request_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            task_id=task_id,
            client=client,
            request_id=request_id,
            authorization=authorization,
        )
    ).parsed
