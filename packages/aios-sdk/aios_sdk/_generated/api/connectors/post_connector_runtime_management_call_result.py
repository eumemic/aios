from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.runtime_management_call_result_request import (
    RuntimeManagementCallResultRequest,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: RuntimeManagementCallResultRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/runtime/management-call-results",
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
    body: RuntimeManagementCallResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Runtime Management Call Result

     Resolve a pending management call by posting the connector's result.

    Authorization: the runtime bearer's connector must match the
    pending call's connector.  The handler does the scope check, runs
    a conditional UPDATE (only resolves rows still in ``pending``),
    and on success fires the wake NOTIFY on
    ``connector_result_<call_id>`` — the operator's request is LISTENing
    there via :func:`aios.db.listen.listen_for_connector_result`.

    Args:
        authorization (None | str | Unset):
        body (RuntimeManagementCallResultRequest): Body for ``POST
            /v1/connectors/runtime/management-call-results``.

            The runtime container POSTs this after dispatching a management call
            received via the ``/runtime/management-calls`` SSE.  Idempotent on
            ``call_id`` — a second POST whose row has already moved out of
            ``pending`` no-ops (no double-NOTIFY, so the operator can't get two
            wakes).

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
    body: RuntimeManagementCallResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Runtime Management Call Result

     Resolve a pending management call by posting the connector's result.

    Authorization: the runtime bearer's connector must match the
    pending call's connector.  The handler does the scope check, runs
    a conditional UPDATE (only resolves rows still in ``pending``),
    and on success fires the wake NOTIFY on
    ``connector_result_<call_id>`` — the operator's request is LISTENing
    there via :func:`aios.db.listen.listen_for_connector_result`.

    Args:
        authorization (None | str | Unset):
        body (RuntimeManagementCallResultRequest): Body for ``POST
            /v1/connectors/runtime/management-call-results``.

            The runtime container POSTs this after dispatching a management call
            received via the ``/runtime/management-calls`` SSE.  Idempotent on
            ``call_id`` — a second POST whose row has already moved out of
            ``pending`` no-ops (no double-NOTIFY, so the operator can't get two
            wakes).

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
    body: RuntimeManagementCallResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Runtime Management Call Result

     Resolve a pending management call by posting the connector's result.

    Authorization: the runtime bearer's connector must match the
    pending call's connector.  The handler does the scope check, runs
    a conditional UPDATE (only resolves rows still in ``pending``),
    and on success fires the wake NOTIFY on
    ``connector_result_<call_id>`` — the operator's request is LISTENing
    there via :func:`aios.db.listen.listen_for_connector_result`.

    Args:
        authorization (None | str | Unset):
        body (RuntimeManagementCallResultRequest): Body for ``POST
            /v1/connectors/runtime/management-call-results``.

            The runtime container POSTs this after dispatching a management call
            received via the ``/runtime/management-calls`` SSE.  Idempotent on
            ``call_id`` — a second POST whose row has already moved out of
            ``pending`` no-ops (no double-NOTIFY, so the operator can't get two
            wakes).

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
    body: RuntimeManagementCallResultRequest,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Runtime Management Call Result

     Resolve a pending management call by posting the connector's result.

    Authorization: the runtime bearer's connector must match the
    pending call's connector.  The handler does the scope check, runs
    a conditional UPDATE (only resolves rows still in ``pending``),
    and on success fires the wake NOTIFY on
    ``connector_result_<call_id>`` — the operator's request is LISTENing
    there via :func:`aios.db.listen.listen_for_connector_result`.

    Args:
        authorization (None | str | Unset):
        body (RuntimeManagementCallResultRequest): Body for ``POST
            /v1/connectors/runtime/management-call-results``.

            The runtime container POSTs this after dispatching a management call
            received via the ``/runtime/management-calls`` SSE.  Idempotent on
            ``call_id`` — a second POST whose row has already moved out of
            ``pending`` no-ops (no double-NOTIFY, so the operator can't get two
            wakes).

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
