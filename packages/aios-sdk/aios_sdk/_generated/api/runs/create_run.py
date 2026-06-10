from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.wf_run import WfRun
from ...models.wf_run_create import WfRunCreate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: WfRunCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/runs",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WfRun | None:
    if response.status_code == 201:
        response_201 = WfRun.from_dict(response.json())

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
) -> Response[HTTPValidationError | WfRun]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: WfRunCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRun]:
    """Create Run

     Launch a run of a workflow. Snapshots the workflow's current script, binds
    the run to ``environment_id`` (its ``agent()`` children spawn there) and to
    ``vault_ids`` (credentials it resolves at tool-call time), and wakes it. A missing
    workflow or environment 404s. The HTTP path is unattenuated operator authority — no
    ``launcher_session_id``, so the requested vaults are bound as-is (account-scoped).

    Args:
        authorization (None | str | Unset):
        body (WfRunCreate): Request body for ``POST /v1/runs`` — launch a run of a workflow.

            ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
            binds to ``environment_id`` (like a session), into which its ``agent()`` children
            spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
            ride in request bodies; the HTTP path is always an operator launch.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRun]
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
    body: WfRunCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRun | None:
    """Create Run

     Launch a run of a workflow. Snapshots the workflow's current script, binds
    the run to ``environment_id`` (its ``agent()`` children spawn there) and to
    ``vault_ids`` (credentials it resolves at tool-call time), and wakes it. A missing
    workflow or environment 404s. The HTTP path is unattenuated operator authority — no
    ``launcher_session_id``, so the requested vaults are bound as-is (account-scoped).

    Args:
        authorization (None | str | Unset):
        body (WfRunCreate): Request body for ``POST /v1/runs`` — launch a run of a workflow.

            ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
            binds to ``environment_id`` (like a session), into which its ``agent()`` children
            spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
            ride in request bodies; the HTTP path is always an operator launch.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRun
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: WfRunCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRun]:
    """Create Run

     Launch a run of a workflow. Snapshots the workflow's current script, binds
    the run to ``environment_id`` (its ``agent()`` children spawn there) and to
    ``vault_ids`` (credentials it resolves at tool-call time), and wakes it. A missing
    workflow or environment 404s. The HTTP path is unattenuated operator authority — no
    ``launcher_session_id``, so the requested vaults are bound as-is (account-scoped).

    Args:
        authorization (None | str | Unset):
        body (WfRunCreate): Request body for ``POST /v1/runs`` — launch a run of a workflow.

            ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
            binds to ``environment_id`` (like a session), into which its ``agent()`` children
            spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
            ride in request bodies; the HTTP path is always an operator launch.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRun]
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
    body: WfRunCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRun | None:
    """Create Run

     Launch a run of a workflow. Snapshots the workflow's current script, binds
    the run to ``environment_id`` (its ``agent()`` children spawn there) and to
    ``vault_ids`` (credentials it resolves at tool-call time), and wakes it. A missing
    workflow or environment 404s. The HTTP path is unattenuated operator authority — no
    ``launcher_session_id``, so the requested vaults are bound as-is (account-scoped).

    Args:
        authorization (None | str | Unset):
        body (WfRunCreate): Request body for ``POST /v1/runs`` — launch a run of a workflow.

            ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
            binds to ``environment_id`` (like a session), into which its ``agent()`` children
            spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
            ride in request bodies; the HTTP path is always an operator launch.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRun
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
