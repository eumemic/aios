from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.input_batch import InputBatch
from ...types import UNSET, Response, Unset


def _get_kwargs(
    grant_id: str,
    *,
    body: InputBatch,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/browser/takeover/{grant_id}/input".format(
            grant_id=quote(str(grant_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

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
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: InputBatch,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Input

     Append one epoch-stamped input batch to the shared-filesystem spool.

    No worker involvement. The check-then-append race (grant closes between
    the epoch check and the write) is harmless — the driver drops stale-epoch
    lines, being the enforcement authority. 409 on a closed grant or stale
    epoch; 413 when the spool would exceed its byte cap.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (InputBatch): One coalesced batch of input events, epoch-stamped.

            The API pre-checks the epoch against the grant record; the DRIVER is the
            enforcement authority and drops stale-epoch spool lines regardless.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: InputBatch,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Input

     Append one epoch-stamped input batch to the shared-filesystem spool.

    No worker involvement. The check-then-append race (grant closes between
    the epoch check and the write) is harmless — the driver drops stale-epoch
    lines, being the enforcement authority. 409 on a closed grant or stale
    epoch; 413 when the spool would exceed its byte cap.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (InputBatch): One coalesced batch of input events, epoch-stamped.

            The API pre-checks the epoch against the grant record; the DRIVER is the
            enforcement authority and drops stale-epoch spool lines regardless.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        grant_id=grant_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: InputBatch,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Input

     Append one epoch-stamped input batch to the shared-filesystem spool.

    No worker involvement. The check-then-append race (grant closes between
    the epoch check and the write) is harmless — the driver drops stale-epoch
    lines, being the enforcement authority. 409 on a closed grant or stale
    epoch; 413 when the spool would exceed its byte cap.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (InputBatch): One coalesced batch of input events, epoch-stamped.

            The API pre-checks the epoch against the grant record; the DRIVER is the
            enforcement authority and drops stale-epoch spool lines regardless.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    grant_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: InputBatch,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Input

     Append one epoch-stamped input batch to the shared-filesystem spool.

    No worker involvement. The check-then-append race (grant closes between
    the epoch check and the write) is harmless — the driver drops stale-epoch
    lines, being the enforcement authority. 409 on a closed grant or stale
    epoch; 413 when the spool would exceed its byte cap.

    Args:
        grant_id (str):
        authorization (None | str | Unset):
        body (InputBatch): One coalesced batch of input events, epoch-stamped.

            The API pre-checks the epoch against the grant record; the DRIVER is the
            enforcement authority and drops stale-epoch spool lines regardless.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            grant_id=grant_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
