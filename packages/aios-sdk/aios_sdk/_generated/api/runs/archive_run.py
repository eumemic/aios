from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.wf_run import WfRun
from ...types import UNSET, Response, Unset


def _get_kwargs(
    run_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/runs/{run_id}/archive".format(
            run_id=quote(str(run_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WfRun | None:
    if response.status_code == 200:
        response_200 = WfRun.from_dict(response.json())

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
) -> Response[HTTPValidationError | WfRun]:
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
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRun]:
    """Archive Run

     Archive a TERMINAL run. Archived runs disappear from the default list_runs
    but remain fetchable by id and keep their journal — the run-side analog of
    archive_workflow. A non-terminal run (pending/running/suspended) is refused with
    409 Conflict; an already-archived run is an idempotent 409. There is no
    unarchive — terminal+archived is a final state.

    Args:
        run_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRun]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
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
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRun | None:
    """Archive Run

     Archive a TERMINAL run. Archived runs disappear from the default list_runs
    but remain fetchable by id and keep their journal — the run-side analog of
    archive_workflow. A non-terminal run (pending/running/suspended) is refused with
    409 Conflict; an already-archived run is an idempotent 409. There is no
    unarchive — terminal+archived is a final state.

    Args:
        run_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRun
    """

    return sync_detailed(
        run_id=run_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRun]:
    """Archive Run

     Archive a TERMINAL run. Archived runs disappear from the default list_runs
    but remain fetchable by id and keep their journal — the run-side analog of
    archive_workflow. A non-terminal run (pending/running/suspended) is refused with
    409 Conflict; an already-archived run is an idempotent 409. There is no
    unarchive — terminal+archived is a final state.

    Args:
        run_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRun]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRun | None:
    """Archive Run

     Archive a TERMINAL run. Archived runs disappear from the default list_runs
    but remain fetchable by id and keep their journal — the run-side analog of
    archive_workflow. A non-terminal run (pending/running/suspended) is refused with
    409 Conflict; an already-archived run is an idempotent 409. There is no
    unarchive — terminal+archived is a final state.

    Args:
        run_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRun
    """

    return (
        await asyncio_detailed(
            run_id=run_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
