from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    target_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/accounts/{target_id}/purge".format(
            target_id=quote(str(target_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

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
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Purge Account

     Hard-delete a direct child that has already been soft-archived.

    T2 decision (#1463): ``purge`` is deliberately retained as the
    explicitly-named two-step hard-delete ceremony (it refuses unless the
    account is already archived, childless, and resourceless). It is *not*
    renamed to ``delete_account`` — under the T2 convention the bare DELETE
    verb is the soft-archive, and ``/purge`` is the consistent name for the
    irreversible hard-delete across families. accounts is the baseline here.

    Two-step ceremony:     1. ``DELETE /v1/accounts/{id}`` soft-archives (sets ``archived_at``).    2.
    ``POST /v1/accounts/{id}/purge`` hard-deletes the row.

    Refuses with 409 if the account is not yet archived, has non-archived
    children, has any resources (FK RESTRICT will refuse the DELETE), or
    is the caller's own account. Compliance / GDPR path; the normal
    lifecycle stops at archive.

    Args:
        target_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        target_id=target_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Purge Account

     Hard-delete a direct child that has already been soft-archived.

    T2 decision (#1463): ``purge`` is deliberately retained as the
    explicitly-named two-step hard-delete ceremony (it refuses unless the
    account is already archived, childless, and resourceless). It is *not*
    renamed to ``delete_account`` — under the T2 convention the bare DELETE
    verb is the soft-archive, and ``/purge`` is the consistent name for the
    irreversible hard-delete across families. accounts is the baseline here.

    Two-step ceremony:     1. ``DELETE /v1/accounts/{id}`` soft-archives (sets ``archived_at``).    2.
    ``POST /v1/accounts/{id}/purge`` hard-deletes the row.

    Refuses with 409 if the account is not yet archived, has non-archived
    children, has any resources (FK RESTRICT will refuse the DELETE), or
    is the caller's own account. Compliance / GDPR path; the normal
    lifecycle stops at archive.

    Args:
        target_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        target_id=target_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Purge Account

     Hard-delete a direct child that has already been soft-archived.

    T2 decision (#1463): ``purge`` is deliberately retained as the
    explicitly-named two-step hard-delete ceremony (it refuses unless the
    account is already archived, childless, and resourceless). It is *not*
    renamed to ``delete_account`` — under the T2 convention the bare DELETE
    verb is the soft-archive, and ``/purge`` is the consistent name for the
    irreversible hard-delete across families. accounts is the baseline here.

    Two-step ceremony:     1. ``DELETE /v1/accounts/{id}`` soft-archives (sets ``archived_at``).    2.
    ``POST /v1/accounts/{id}/purge`` hard-deletes the row.

    Refuses with 409 if the account is not yet archived, has non-archived
    children, has any resources (FK RESTRICT will refuse the DELETE), or
    is the caller's own account. Compliance / GDPR path; the normal
    lifecycle stops at archive.

    Args:
        target_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        target_id=target_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Purge Account

     Hard-delete a direct child that has already been soft-archived.

    T2 decision (#1463): ``purge`` is deliberately retained as the
    explicitly-named two-step hard-delete ceremony (it refuses unless the
    account is already archived, childless, and resourceless). It is *not*
    renamed to ``delete_account`` — under the T2 convention the bare DELETE
    verb is the soft-archive, and ``/purge`` is the consistent name for the
    irreversible hard-delete across families. accounts is the baseline here.

    Two-step ceremony:     1. ``DELETE /v1/accounts/{id}`` soft-archives (sets ``archived_at``).    2.
    ``POST /v1/accounts/{id}/purge`` hard-deletes the row.

    Refuses with 409 if the account is not yet archived, has non-archived
    children, has any resources (FK RESTRICT will refuse the DELETE), or
    is the caller's own account. Compliance / GDPR path; the normal
    lifecycle stops at archive.

    Args:
        target_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            target_id=target_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
