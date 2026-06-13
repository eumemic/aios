from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.github_repository_resource import GithubRepositoryResource
from ...models.github_repository_resource_echo import GithubRepositoryResourceEcho
from ...models.http_validation_error import HTTPValidationError
from ...models.memory_store_resource import MemoryStoreResource
from ...models.memory_store_resource_echo import MemoryStoreResourceEcho
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: GithubRepositoryResource | MemoryStoreResource,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/resources".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    if isinstance(body, MemoryStoreResource):
        _kwargs["json"] = body.to_dict()
    else:
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError | None
):
    if response.status_code == 201:

        def _parse_response_201(
            data: object,
        ) -> GithubRepositoryResourceEcho | MemoryStoreResourceEcho:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_0 = MemoryStoreResourceEcho.from_dict(data)

                return response_201_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_201_type_1 = GithubRepositoryResourceEcho.from_dict(data)

            return response_201_type_1

        response_201 = _parse_response_201(response.json())

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
) -> Response[
    GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError
]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GithubRepositoryResource | MemoryStoreResource,
    authorization: None | str | Unset = UNSET,
) -> Response[
    GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError
]:
    """Add Resource

     Attach a single resource. Granular add-one operation per #270 —
    additive, so it leaves every other attached resource untouched
    (unlike ``PUT /v1/sessions/{id}`` with ``resources``, which replaces
    the whole list). Dispatches on the body's ``type`` discriminator.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (GithubRepositoryResource | MemoryStoreResource):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GithubRepositoryResource | MemoryStoreResource,
    authorization: None | str | Unset = UNSET,
) -> (
    GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError | None
):
    """Add Resource

     Attach a single resource. Granular add-one operation per #270 —
    additive, so it leaves every other attached resource untouched
    (unlike ``PUT /v1/sessions/{id}`` with ``resources``, which replaces
    the whole list). Dispatches on the body's ``type`` discriminator.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (GithubRepositoryResource | MemoryStoreResource):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GithubRepositoryResource | MemoryStoreResource,
    authorization: None | str | Unset = UNSET,
) -> Response[
    GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError
]:
    """Add Resource

     Attach a single resource. Granular add-one operation per #270 —
    additive, so it leaves every other attached resource untouched
    (unlike ``PUT /v1/sessions/{id}`` with ``resources``, which replaces
    the whole list). Dispatches on the body's ``type`` discriminator.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (GithubRepositoryResource | MemoryStoreResource):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GithubRepositoryResource | MemoryStoreResource,
    authorization: None | str | Unset = UNSET,
) -> (
    GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError | None
):
    """Add Resource

     Attach a single resource. Granular add-one operation per #270 —
    additive, so it leaves every other attached resource untouched
    (unlike ``PUT /v1/sessions/{id}`` with ``resources``, which replaces
    the whole list). Dispatches on the body's ``type`` discriminator.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (GithubRepositoryResource | MemoryStoreResource):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GithubRepositoryResourceEcho | MemoryStoreResourceEcho | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
