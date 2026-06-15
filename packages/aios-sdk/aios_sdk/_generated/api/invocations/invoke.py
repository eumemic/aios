from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.invocation_handle import InvocationHandle
from ...models.invocation_request import InvocationRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: InvocationRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/invocations",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | InvocationHandle | None:
    if response.status_code == 201:
        response_201 = InvocationHandle.from_dict(response.json())

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
) -> Response[HTTPValidationError | InvocationHandle]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: InvocationRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | InvocationHandle]:
    """Invoke

     Write a trusted request edge + resolve-or-create a servicer, returning a handle.

    ``target_kind=agent`` creates a **session** servicer (env-bound) and injects a
    channel-less request; ``target_kind=workflow`` creates a **run** servicer;
    ``target_kind=session`` invokes an existing same-account session by id. The
    returned ``request_id`` correlates the matching awaiter
    (``GET /v1/sessions/{id}/await?request_id=`` for sessions, ``GET /v1/runs/{id}/wait``
    for runs). A cross-tenant ``target`` 404s before any edge is written; a supplied
    ``environment_id`` is ownership-checked against the caller's account.

    Args:
        authorization (None | str | Unset):
        body (InvocationRequest): Request body for ``POST /v1/invocations`` — the API caller's
            request-writer.

            ``target`` is an ``agent_id | workflow_id | session_id`` and ``target_kind``
            discriminates it:

            * ``agent``     — create a **session** servicer and inject a channel-less
              request into it (the API analog of ``invoke_agent``).
            * ``workflow``  — create a **run** servicer of the workflow.
            * ``session``   — invoke an **existing** same-account session by id (the API
              analog of #1127's ``invoke(session_id)``). No ``environment_id`` applies —
              the session already exists.

            ``output_schema`` is the per-request JSON Schema the response ``value`` must
            satisfy; it rides the edge (``metadata.request.output_schema``), coexisting
            with any definition-level schema. ``environment_id`` is ownership-checked
            against the caller's account on the ``agent`` / ``workflow`` create-paths
            (the per-field containment clamp is #1130's deliverable).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | InvocationHandle]
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
    body: InvocationRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | InvocationHandle | None:
    """Invoke

     Write a trusted request edge + resolve-or-create a servicer, returning a handle.

    ``target_kind=agent`` creates a **session** servicer (env-bound) and injects a
    channel-less request; ``target_kind=workflow`` creates a **run** servicer;
    ``target_kind=session`` invokes an existing same-account session by id. The
    returned ``request_id`` correlates the matching awaiter
    (``GET /v1/sessions/{id}/await?request_id=`` for sessions, ``GET /v1/runs/{id}/wait``
    for runs). A cross-tenant ``target`` 404s before any edge is written; a supplied
    ``environment_id`` is ownership-checked against the caller's account.

    Args:
        authorization (None | str | Unset):
        body (InvocationRequest): Request body for ``POST /v1/invocations`` — the API caller's
            request-writer.

            ``target`` is an ``agent_id | workflow_id | session_id`` and ``target_kind``
            discriminates it:

            * ``agent``     — create a **session** servicer and inject a channel-less
              request into it (the API analog of ``invoke_agent``).
            * ``workflow``  — create a **run** servicer of the workflow.
            * ``session``   — invoke an **existing** same-account session by id (the API
              analog of #1127's ``invoke(session_id)``). No ``environment_id`` applies —
              the session already exists.

            ``output_schema`` is the per-request JSON Schema the response ``value`` must
            satisfy; it rides the edge (``metadata.request.output_schema``), coexisting
            with any definition-level schema. ``environment_id`` is ownership-checked
            against the caller's account on the ``agent`` / ``workflow`` create-paths
            (the per-field containment clamp is #1130's deliverable).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | InvocationHandle
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: InvocationRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | InvocationHandle]:
    """Invoke

     Write a trusted request edge + resolve-or-create a servicer, returning a handle.

    ``target_kind=agent`` creates a **session** servicer (env-bound) and injects a
    channel-less request; ``target_kind=workflow`` creates a **run** servicer;
    ``target_kind=session`` invokes an existing same-account session by id. The
    returned ``request_id`` correlates the matching awaiter
    (``GET /v1/sessions/{id}/await?request_id=`` for sessions, ``GET /v1/runs/{id}/wait``
    for runs). A cross-tenant ``target`` 404s before any edge is written; a supplied
    ``environment_id`` is ownership-checked against the caller's account.

    Args:
        authorization (None | str | Unset):
        body (InvocationRequest): Request body for ``POST /v1/invocations`` — the API caller's
            request-writer.

            ``target`` is an ``agent_id | workflow_id | session_id`` and ``target_kind``
            discriminates it:

            * ``agent``     — create a **session** servicer and inject a channel-less
              request into it (the API analog of ``invoke_agent``).
            * ``workflow``  — create a **run** servicer of the workflow.
            * ``session``   — invoke an **existing** same-account session by id (the API
              analog of #1127's ``invoke(session_id)``). No ``environment_id`` applies —
              the session already exists.

            ``output_schema`` is the per-request JSON Schema the response ``value`` must
            satisfy; it rides the edge (``metadata.request.output_schema``), coexisting
            with any definition-level schema. ``environment_id`` is ownership-checked
            against the caller's account on the ``agent`` / ``workflow`` create-paths
            (the per-field containment clamp is #1130's deliverable).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | InvocationHandle]
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
    body: InvocationRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | InvocationHandle | None:
    """Invoke

     Write a trusted request edge + resolve-or-create a servicer, returning a handle.

    ``target_kind=agent`` creates a **session** servicer (env-bound) and injects a
    channel-less request; ``target_kind=workflow`` creates a **run** servicer;
    ``target_kind=session`` invokes an existing same-account session by id. The
    returned ``request_id`` correlates the matching awaiter
    (``GET /v1/sessions/{id}/await?request_id=`` for sessions, ``GET /v1/runs/{id}/wait``
    for runs). A cross-tenant ``target`` 404s before any edge is written; a supplied
    ``environment_id`` is ownership-checked against the caller's account.

    Args:
        authorization (None | str | Unset):
        body (InvocationRequest): Request body for ``POST /v1/invocations`` — the API caller's
            request-writer.

            ``target`` is an ``agent_id | workflow_id | session_id`` and ``target_kind``
            discriminates it:

            * ``agent``     — create a **session** servicer and inject a channel-less
              request into it (the API analog of ``invoke_agent``).
            * ``workflow``  — create a **run** servicer of the workflow.
            * ``session``   — invoke an **existing** same-account session by id (the API
              analog of #1127's ``invoke(session_id)``). No ``environment_id`` applies —
              the session already exists.

            ``output_schema`` is the per-request JSON Schema the response ``value`` must
            satisfy; it rides the edge (``metadata.request.output_schema``), coexisting
            with any definition-level schema. ``environment_id`` is ownership-checked
            against the caller's account on the ``agent`` / ``workflow`` create-paths
            (the per-field containment clamp is #1130's deliverable).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | InvocationHandle
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
