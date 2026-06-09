from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.workflow import Workflow
from ...models.workflow_update import WorkflowUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    workflow_id: str,
    *,
    body: WorkflowUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/workflows/{workflow_id}".format(
            workflow_id=quote(str(workflow_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Workflow | None:
    if response.status_code == 200:
        response_200 = Workflow.from_dict(response.json())

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
) -> Response[HTTPValidationError | Workflow]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    workflow_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: WorkflowUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Workflow]:
    """Update Workflow

     Update a workflow in place, bumping ``version``.

    ``body.version`` must match the current version (optimistic concurrency — 409 on a
    stale token; re-fetch and retry). Omitted fields are preserved; an identical update
    is a no-op. In-flight runs are unaffected (a run snapshots script + surface at
    launch). The HTTP path is unattenuated operator authority — no ``actor_session_id``.

    Args:
        workflow_id (str):
        authorization (None | str | Unset):
        body (WorkflowUpdate): Request body for ``PUT /v1/workflows/{id}`` — update in place,
            bumping ``version``.

            ``version`` is the optimistic-concurrency token: it must match the workflow's
            current version or the update 409s (re-fetch and retry). Omitted fields are
            preserved — nullable fields (``input_schema``/``output_schema``/``description``)
            can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
            An identical update is a no-op (no bump). There is no version-snapshot table —
            a run pins ``script`` + the declared surface onto itself at launch, so in-flight
            runs never observe an update. (The ``AgentUpdate`` shape, minus history.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Workflow]
    """

    kwargs = _get_kwargs(
        workflow_id=workflow_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    workflow_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: WorkflowUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Workflow | None:
    """Update Workflow

     Update a workflow in place, bumping ``version``.

    ``body.version`` must match the current version (optimistic concurrency — 409 on a
    stale token; re-fetch and retry). Omitted fields are preserved; an identical update
    is a no-op. In-flight runs are unaffected (a run snapshots script + surface at
    launch). The HTTP path is unattenuated operator authority — no ``actor_session_id``.

    Args:
        workflow_id (str):
        authorization (None | str | Unset):
        body (WorkflowUpdate): Request body for ``PUT /v1/workflows/{id}`` — update in place,
            bumping ``version``.

            ``version`` is the optimistic-concurrency token: it must match the workflow's
            current version or the update 409s (re-fetch and retry). Omitted fields are
            preserved — nullable fields (``input_schema``/``output_schema``/``description``)
            can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
            An identical update is a no-op (no bump). There is no version-snapshot table —
            a run pins ``script`` + the declared surface onto itself at launch, so in-flight
            runs never observe an update. (The ``AgentUpdate`` shape, minus history.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Workflow
    """

    return sync_detailed(
        workflow_id=workflow_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    workflow_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: WorkflowUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Workflow]:
    """Update Workflow

     Update a workflow in place, bumping ``version``.

    ``body.version`` must match the current version (optimistic concurrency — 409 on a
    stale token; re-fetch and retry). Omitted fields are preserved; an identical update
    is a no-op. In-flight runs are unaffected (a run snapshots script + surface at
    launch). The HTTP path is unattenuated operator authority — no ``actor_session_id``.

    Args:
        workflow_id (str):
        authorization (None | str | Unset):
        body (WorkflowUpdate): Request body for ``PUT /v1/workflows/{id}`` — update in place,
            bumping ``version``.

            ``version`` is the optimistic-concurrency token: it must match the workflow's
            current version or the update 409s (re-fetch and retry). Omitted fields are
            preserved — nullable fields (``input_schema``/``output_schema``/``description``)
            can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
            An identical update is a no-op (no bump). There is no version-snapshot table —
            a run pins ``script`` + the declared surface onto itself at launch, so in-flight
            runs never observe an update. (The ``AgentUpdate`` shape, minus history.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Workflow]
    """

    kwargs = _get_kwargs(
        workflow_id=workflow_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    workflow_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: WorkflowUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Workflow | None:
    """Update Workflow

     Update a workflow in place, bumping ``version``.

    ``body.version`` must match the current version (optimistic concurrency — 409 on a
    stale token; re-fetch and retry). Omitted fields are preserved; an identical update
    is a no-op. In-flight runs are unaffected (a run snapshots script + surface at
    launch). The HTTP path is unattenuated operator authority — no ``actor_session_id``.

    Args:
        workflow_id (str):
        authorization (None | str | Unset):
        body (WorkflowUpdate): Request body for ``PUT /v1/workflows/{id}`` — update in place,
            bumping ``version``.

            ``version`` is the optimistic-concurrency token: it must match the workflow's
            current version or the update 409s (re-fetch and retry). Omitted fields are
            preserved — nullable fields (``input_schema``/``output_schema``/``description``)
            can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
            An identical update is a no-op (no bump). There is no version-snapshot table —
            a run pins ``script`` + the declared surface onto itself at launch, so in-flight
            runs never observe an update. (The ``AgentUpdate`` shape, minus history.)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Workflow
    """

    return (
        await asyncio_detailed(
            workflow_id=workflow_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
