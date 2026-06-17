from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.trigger_created import TriggerCreated
from ...models.trigger_update import TriggerUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    name: str,
    *,
    body: TriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/sessions/{session_id}/triggers/{name}".format(
            session_id=quote(str(session_id), safe=""),
            name=quote(str(name), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TriggerCreated | None:
    if response.status_code == 200:
        response_200 = TriggerCreated.from_dict(response.json())

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
) -> Response[HTTPValidationError | TriggerCreated]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: TriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TriggerCreated]:
    """Update Trigger

     Replace a trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    A source-replace TO ``external_event`` (or a re-mint of an already-external
    source = rotation) surfaces a fresh ``ingest_token`` once; otherwise
    ``ingest_token`` is ``null``.

    Args:
        session_id (str):
        name (str):
        authorization (None | str | Unset):
        body (TriggerUpdate): Update body. ``source`` / ``action`` are replaced WHOLESALE when
            provided (a cron↔one-shot or sandbox↔wake conversion is just a
            different object) — via the Replace union variants, whose
            optional-at-create fields are required so a partial object 422s instead
            of silently re-defaulting. ``None`` = leave alone; there is no
            clear-to-null (both columns are NOT NULL). The next_fire / cap /
            past-fire_at business rules are enforced in the service layer (§2.4).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TriggerCreated]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        name=name,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: TriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TriggerCreated | None:
    """Update Trigger

     Replace a trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    A source-replace TO ``external_event`` (or a re-mint of an already-external
    source = rotation) surfaces a fresh ``ingest_token`` once; otherwise
    ``ingest_token`` is ``null``.

    Args:
        session_id (str):
        name (str):
        authorization (None | str | Unset):
        body (TriggerUpdate): Update body. ``source`` / ``action`` are replaced WHOLESALE when
            provided (a cron↔one-shot or sandbox↔wake conversion is just a
            different object) — via the Replace union variants, whose
            optional-at-create fields are required so a partial object 422s instead
            of silently re-defaulting. ``None`` = leave alone; there is no
            clear-to-null (both columns are NOT NULL). The next_fire / cap /
            past-fire_at business rules are enforced in the service layer (§2.4).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TriggerCreated
    """

    return sync_detailed(
        session_id=session_id,
        name=name,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: TriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TriggerCreated]:
    """Update Trigger

     Replace a trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    A source-replace TO ``external_event`` (or a re-mint of an already-external
    source = rotation) surfaces a fresh ``ingest_token`` once; otherwise
    ``ingest_token`` is ``null``.

    Args:
        session_id (str):
        name (str):
        authorization (None | str | Unset):
        body (TriggerUpdate): Update body. ``source`` / ``action`` are replaced WHOLESALE when
            provided (a cron↔one-shot or sandbox↔wake conversion is just a
            different object) — via the Replace union variants, whose
            optional-at-create fields are required so a partial object 422s instead
            of silently re-defaulting. ``None`` = leave alone; there is no
            clear-to-null (both columns are NOT NULL). The next_fire / cap /
            past-fire_at business rules are enforced in the service layer (§2.4).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TriggerCreated]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        name=name,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: TriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TriggerCreated | None:
    """Update Trigger

     Replace a trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    A source-replace TO ``external_event`` (or a re-mint of an already-external
    source = rotation) surfaces a fresh ``ingest_token`` once; otherwise
    ``ingest_token`` is ``null``.

    Args:
        session_id (str):
        name (str):
        authorization (None | str | Unset):
        body (TriggerUpdate): Update body. ``source`` / ``action`` are replaced WHOLESALE when
            provided (a cron↔one-shot or sandbox↔wake conversion is just a
            different object) — via the Replace union variants, whose
            optional-at-create fields are required so a partial object 422s instead
            of silently re-defaulting. ``None`` = leave alone; there is no
            clear-to-null (both columns are NOT NULL). The next_fire / cap /
            past-fire_at business rules are enforced in the service layer (§2.4).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TriggerCreated
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            name=name,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
