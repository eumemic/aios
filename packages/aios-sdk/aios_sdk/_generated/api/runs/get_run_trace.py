from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.trace_response import TraceResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    run_id: str,
    *,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["verbose"] = verbose

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/runs/{run_id}/trace".format(
            run_id=quote(str(run_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TraceResponse | None:
    if response.status_code == 200:
        response_200 = TraceResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | TraceResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TraceResponse]:
    """Get Run Trace

     One-call linear trace of a run + all nested sessions and sub-runs (#1149).

    A read-projection over the invoke-edge tree: walks the parent→child edge
    tree from this run, normalizes each node's outcome to
    ``terminal_state ∈ {ok,errored,cancelled,suspended,running}`` (+ raw
    ``error_kind``), and interleaves the nodes' journals into a flat
    **DFS-pre-order** list (a CLI renders ``depth`` as indentation). A
    deliberately-failed nested session's death cause is visible at a glance — the
    node carries ``terminal_state`` + ``error_kind`` (e.g. ``no_return``) and the
    abbreviated default co-locates the proximate ``is_error`` frame.

    ``verbose=false`` (default) keeps only the load-bearing frames per node
    (request/response/turn lifecycle, gates, errors); ``verbose=true`` lifts the
    filter to the full per-node firehose. The walk runs in one ``REPEATABLE
    READ`` snapshot with a node-count ceiling; a tree past the ceiling returns a
    typed ``truncated: {at_nodes}`` marker.

    Ordering caveat: cross-subtree time-ordering is best-effort to transaction
    granularity; only the causal parent→child edge is exact. ``wake_session``
    peer-pokes are out of scope (a peer stimulus, not a spawn). A cross-tenant
    run 404s.

    Args:
        run_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TraceResponse]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        verbose=verbose,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TraceResponse | None:
    """Get Run Trace

     One-call linear trace of a run + all nested sessions and sub-runs (#1149).

    A read-projection over the invoke-edge tree: walks the parent→child edge
    tree from this run, normalizes each node's outcome to
    ``terminal_state ∈ {ok,errored,cancelled,suspended,running}`` (+ raw
    ``error_kind``), and interleaves the nodes' journals into a flat
    **DFS-pre-order** list (a CLI renders ``depth`` as indentation). A
    deliberately-failed nested session's death cause is visible at a glance — the
    node carries ``terminal_state`` + ``error_kind`` (e.g. ``no_return``) and the
    abbreviated default co-locates the proximate ``is_error`` frame.

    ``verbose=false`` (default) keeps only the load-bearing frames per node
    (request/response/turn lifecycle, gates, errors); ``verbose=true`` lifts the
    filter to the full per-node firehose. The walk runs in one ``REPEATABLE
    READ`` snapshot with a node-count ceiling; a tree past the ceiling returns a
    typed ``truncated: {at_nodes}`` marker.

    Ordering caveat: cross-subtree time-ordering is best-effort to transaction
    granularity; only the causal parent→child edge is exact. ``wake_session``
    peer-pokes are out of scope (a peer stimulus, not a spawn). A cross-tenant
    run 404s.

    Args:
        run_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TraceResponse
    """

    return sync_detailed(
        run_id=run_id,
        client=client,
        verbose=verbose,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TraceResponse]:
    """Get Run Trace

     One-call linear trace of a run + all nested sessions and sub-runs (#1149).

    A read-projection over the invoke-edge tree: walks the parent→child edge
    tree from this run, normalizes each node's outcome to
    ``terminal_state ∈ {ok,errored,cancelled,suspended,running}`` (+ raw
    ``error_kind``), and interleaves the nodes' journals into a flat
    **DFS-pre-order** list (a CLI renders ``depth`` as indentation). A
    deliberately-failed nested session's death cause is visible at a glance — the
    node carries ``terminal_state`` + ``error_kind`` (e.g. ``no_return``) and the
    abbreviated default co-locates the proximate ``is_error`` frame.

    ``verbose=false`` (default) keeps only the load-bearing frames per node
    (request/response/turn lifecycle, gates, errors); ``verbose=true`` lifts the
    filter to the full per-node firehose. The walk runs in one ``REPEATABLE
    READ`` snapshot with a node-count ceiling; a tree past the ceiling returns a
    typed ``truncated: {at_nodes}`` marker.

    Ordering caveat: cross-subtree time-ordering is best-effort to transaction
    granularity; only the causal parent→child edge is exact. ``wake_session``
    peer-pokes are out of scope (a peer stimulus, not a spawn). A cross-tenant
    run 404s.

    Args:
        run_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TraceResponse]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        verbose=verbose,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TraceResponse | None:
    """Get Run Trace

     One-call linear trace of a run + all nested sessions and sub-runs (#1149).

    A read-projection over the invoke-edge tree: walks the parent→child edge
    tree from this run, normalizes each node's outcome to
    ``terminal_state ∈ {ok,errored,cancelled,suspended,running}`` (+ raw
    ``error_kind``), and interleaves the nodes' journals into a flat
    **DFS-pre-order** list (a CLI renders ``depth`` as indentation). A
    deliberately-failed nested session's death cause is visible at a glance — the
    node carries ``terminal_state`` + ``error_kind`` (e.g. ``no_return``) and the
    abbreviated default co-locates the proximate ``is_error`` frame.

    ``verbose=false`` (default) keeps only the load-bearing frames per node
    (request/response/turn lifecycle, gates, errors); ``verbose=true`` lifts the
    filter to the full per-node firehose. The walk runs in one ``REPEATABLE
    READ`` snapshot with a node-count ceiling; a tree past the ceiling returns a
    typed ``truncated: {at_nodes}`` marker.

    Ordering caveat: cross-subtree time-ordering is best-effort to transaction
    granularity; only the causal parent→child edge is exact. ``wake_session``
    peer-pokes are out of scope (a peer stimulus, not a spawn). A cross-tenant
    run 404s.

    Args:
        run_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TraceResponse
    """

    return (
        await asyncio_detailed(
            run_id=run_id,
            client=client,
            verbose=verbose,
            authorization=authorization,
        )
    ).parsed
