from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.operator_trigger_echo import OperatorTriggerEcho
from ...models.operator_trigger_update import OperatorTriggerUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    name: str,
    *,
    body: OperatorTriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/triggers/{name}".format(
            name=quote(str(name), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | OperatorTriggerEcho | None:
    if response.status_code == 200:
        response_200 = OperatorTriggerEcho.from_dict(response.json())

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
) -> Response[HTTPValidationError | OperatorTriggerEcho]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: OperatorTriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | OperatorTriggerEcho]:
    """Update Operator Trigger

     Replace an operator trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    Args:
        name (str):
        authorization (None | str | Unset):
        body (OperatorTriggerUpdate): Update body for ``PUT /v1/triggers/{name}``: the same
            Replace semantics as
            :class:`TriggerUpdate`, limited to the operator shapes.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OperatorTriggerEcho]
    """

    kwargs = _get_kwargs(
        name=name,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: OperatorTriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | OperatorTriggerEcho | None:
    """Update Operator Trigger

     Replace an operator trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    Args:
        name (str):
        authorization (None | str | Unset):
        body (OperatorTriggerUpdate): Update body for ``PUT /v1/triggers/{name}``: the same
            Replace semantics as
            :class:`TriggerUpdate`, limited to the operator shapes.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OperatorTriggerEcho
    """

    return sync_detailed(
        name=name,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: OperatorTriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | OperatorTriggerEcho]:
    """Update Operator Trigger

     Replace an operator trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    Args:
        name (str):
        authorization (None | str | Unset):
        body (OperatorTriggerUpdate): Update body for ``PUT /v1/triggers/{name}``: the same
            Replace semantics as
            :class:`TriggerUpdate`, limited to the operator shapes.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OperatorTriggerEcho]
    """

    kwargs = _get_kwargs(
        name=name,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    name: str,
    *,
    client: AuthenticatedClient | Client,
    body: OperatorTriggerUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | OperatorTriggerEcho | None:
    """Update Operator Trigger

     Replace an operator trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale.

    Args:
        name (str):
        authorization (None | str | Unset):
        body (OperatorTriggerUpdate): Update body for ``PUT /v1/triggers/{name}``: the same
            Replace semantics as
            :class:`TriggerUpdate`, limited to the operator shapes.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OperatorTriggerEcho
    """

    return (
        await asyncio_detailed(
            name=name,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
