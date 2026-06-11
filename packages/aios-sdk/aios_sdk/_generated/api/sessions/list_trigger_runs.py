from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_trigger_run_echo import ListResponseTriggerRunEcho
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    name: str,
    *,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/triggers/{name}/runs".format(
            session_id=quote(str(session_id), safe=""),
            name=quote(str(name), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseTriggerRunEcho | None:
    if response.status_code == 200:
        response_200 = ListResponseTriggerRunEcho.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseTriggerRunEcho]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseTriggerRunEcho]:
    """List Trigger Runs

     List a trigger's fires (the per-fire audit), newest first.

    Keyed by name against the audit table's denormalized columns — NOT the
    live trigger row — so one-shot tombstones and a deleted trigger's history
    stay reachable (the audit outlives its trigger by design). Rows older
    than the retention window are pruned.

    Args:
        session_id (str):
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseTriggerRunEcho]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        name=name,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseTriggerRunEcho | None:
    """List Trigger Runs

     List a trigger's fires (the per-fire audit), newest first.

    Keyed by name against the audit table's denormalized columns — NOT the
    live trigger row — so one-shot tombstones and a deleted trigger's history
    stay reachable (the audit outlives its trigger by design). Rows older
    than the retention window are pruned.

    Args:
        session_id (str):
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseTriggerRunEcho
    """

    return sync_detailed(
        session_id=session_id,
        name=name,
        client=client,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseTriggerRunEcho]:
    """List Trigger Runs

     List a trigger's fires (the per-fire audit), newest first.

    Keyed by name against the audit table's denormalized columns — NOT the
    live trigger row — so one-shot tombstones and a deleted trigger's history
    stay reachable (the audit outlives its trigger by design). Rows older
    than the retention window are pruned.

    Args:
        session_id (str):
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseTriggerRunEcho]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        name=name,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseTriggerRunEcho | None:
    """List Trigger Runs

     List a trigger's fires (the per-fire audit), newest first.

    Keyed by name against the audit table's denormalized columns — NOT the
    live trigger row — so one-shot tombstones and a deleted trigger's history
    stay reachable (the audit outlives its trigger by design). Rows older
    than the retention window are pruned.

    Args:
        session_id (str):
        name (str):
        limit (int | Unset):  Default: 50.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseTriggerRunEcho
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            name=name,
            client=client,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
