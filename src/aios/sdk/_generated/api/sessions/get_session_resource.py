from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.github_repository_resource_echo import GithubRepositoryResourceEcho
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    resource_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/resources/{resource_id}".format(
            session_id=quote(str(session_id), safe=""),
            resource_id=quote(str(resource_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> GithubRepositoryResourceEcho | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = GithubRepositoryResourceEcho.from_dict(response.json())

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
) -> Response[GithubRepositoryResourceEcho | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[GithubRepositoryResourceEcho | HTTPValidationError]:
    """Get Resource

     Fetch a single resource attached to ``session_id`` by its id.

    v1 only supports ``github_repository`` (id prefix ``ghrepo_``) since
    memory store attachments are keyed by ``(session_id, memory_store_id)``
    and don't have a separate attachment id.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GithubRepositoryResourceEcho | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        resource_id=resource_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> GithubRepositoryResourceEcho | HTTPValidationError | None:
    """Get Resource

     Fetch a single resource attached to ``session_id`` by its id.

    v1 only supports ``github_repository`` (id prefix ``ghrepo_``) since
    memory store attachments are keyed by ``(session_id, memory_store_id)``
    and don't have a separate attachment id.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GithubRepositoryResourceEcho | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        resource_id=resource_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[GithubRepositoryResourceEcho | HTTPValidationError]:
    """Get Resource

     Fetch a single resource attached to ``session_id`` by its id.

    v1 only supports ``github_repository`` (id prefix ``ghrepo_``) since
    memory store attachments are keyed by ``(session_id, memory_store_id)``
    and don't have a separate attachment id.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GithubRepositoryResourceEcho | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        resource_id=resource_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    resource_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> GithubRepositoryResourceEcho | HTTPValidationError | None:
    """Get Resource

     Fetch a single resource attached to ``session_id`` by its id.

    v1 only supports ``github_repository`` (id prefix ``ghrepo_``) since
    memory store attachments are keyed by ``(session_id, memory_store_id)``
    and don't have a separate attachment id.

    Args:
        session_id (str):
        resource_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GithubRepositoryResourceEcho | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            resource_id=resource_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
