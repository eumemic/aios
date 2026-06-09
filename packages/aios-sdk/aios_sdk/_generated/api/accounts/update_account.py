from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.account import Account
from ...models.http_validation_error import HTTPValidationError
from ...models.update_account_request import UpdateAccountRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    target_id: str,
    *,
    body: UpdateAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/accounts/{target_id}".format(
            target_id=quote(str(target_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Account | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Account.from_dict(response.json())

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
) -> Response[Account | HTTPValidationError]:
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
    body: UpdateAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Account | HTTPValidationError]:
    """Update Account

     Partial-update ``display_name`` / ``can_mint_children`` on a
    caller-or-direct-child account.

    Omitted fields are preserved. Both fields null is a valid no-op
    that returns the current row.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (UpdateAccountRequest): Body for ``PATCH /v1/accounts/{id}``.

            Partial update: omitted fields are preserved. All fields are optional;
            submitting none is a no-op that returns the account row unchanged.
            ``config`` is *merged* into the stored config (only the keys present in
            the submitted object are written), so setting one config item never
            disturbs the others.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Account | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        target_id=target_id,
        body=body,
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
    body: UpdateAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> Account | HTTPValidationError | None:
    """Update Account

     Partial-update ``display_name`` / ``can_mint_children`` on a
    caller-or-direct-child account.

    Omitted fields are preserved. Both fields null is a valid no-op
    that returns the current row.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (UpdateAccountRequest): Body for ``PATCH /v1/accounts/{id}``.

            Partial update: omitted fields are preserved. All fields are optional;
            submitting none is a no-op that returns the account row unchanged.
            ``config`` is *merged* into the stored config (only the keys present in
            the submitted object are written), so setting one config item never
            disturbs the others.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Account | HTTPValidationError
    """

    return sync_detailed(
        target_id=target_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: UpdateAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Account | HTTPValidationError]:
    """Update Account

     Partial-update ``display_name`` / ``can_mint_children`` on a
    caller-or-direct-child account.

    Omitted fields are preserved. Both fields null is a valid no-op
    that returns the current row.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (UpdateAccountRequest): Body for ``PATCH /v1/accounts/{id}``.

            Partial update: omitted fields are preserved. All fields are optional;
            submitting none is a no-op that returns the account row unchanged.
            ``config`` is *merged* into the stored config (only the keys present in
            the submitted object are written), so setting one config item never
            disturbs the others.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Account | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        target_id=target_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: UpdateAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> Account | HTTPValidationError | None:
    """Update Account

     Partial-update ``display_name`` / ``can_mint_children`` on a
    caller-or-direct-child account.

    Omitted fields are preserved. Both fields null is a valid no-op
    that returns the current row.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (UpdateAccountRequest): Body for ``PATCH /v1/accounts/{id}``.

            Partial update: omitted fields are preserved. All fields are optional;
            submitting none is a no-op that returns the account row unchanged.
            ``config`` is *merged* into the stored config (only the keys present in
            the submitted object are written), so setting one config item never
            disturbs the others.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Account | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            target_id=target_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
