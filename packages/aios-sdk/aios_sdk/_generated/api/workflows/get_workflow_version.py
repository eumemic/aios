from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.workflow_version import WorkflowVersion
from ...types import UNSET, Response, Unset


def _get_kwargs(
    workflow_id: str,
    version: int,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/workflows/{workflow_id}/versions/{version}".format(
            workflow_id=quote(str(workflow_id), safe=""),
            version=quote(str(version), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WorkflowVersion | None:
    if response.status_code == 200:
        response_200 = WorkflowVersion.from_dict(response.json())

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
) -> Response[HTTPValidationError | WorkflowVersion]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    workflow_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WorkflowVersion]:
    """Get Workflow Version

     Fetch one historical version's definition snapshot.

    The snapshot reflects the workflow's definition at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        workflow_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WorkflowVersion]
    """

    kwargs = _get_kwargs(
        workflow_id=workflow_id,
        version=version,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    workflow_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WorkflowVersion | None:
    """Get Workflow Version

     Fetch one historical version's definition snapshot.

    The snapshot reflects the workflow's definition at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        workflow_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WorkflowVersion
    """

    return sync_detailed(
        workflow_id=workflow_id,
        version=version,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    workflow_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WorkflowVersion]:
    """Get Workflow Version

     Fetch one historical version's definition snapshot.

    The snapshot reflects the workflow's definition at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        workflow_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WorkflowVersion]
    """

    kwargs = _get_kwargs(
        workflow_id=workflow_id,
        version=version,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    workflow_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WorkflowVersion | None:
    """Get Workflow Version

     Fetch one historical version's definition snapshot.

    The snapshot reflects the workflow's definition at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        workflow_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WorkflowVersion
    """

    return (
        await asyncio_detailed(
            workflow_id=workflow_id,
            version=version,
            client=client,
            authorization=authorization,
        )
    ).parsed
