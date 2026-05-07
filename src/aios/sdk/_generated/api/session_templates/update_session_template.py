from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.session_template import SessionTemplate
from ...models.session_template_update import SessionTemplateUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    template_id: str,
    *,
    body: SessionTemplateUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/session-templates/{template_id}".format(
            template_id=quote(str(template_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SessionTemplate | None:
    if response.status_code == 200:
        response_200 = SessionTemplate.from_dict(response.json())

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
) -> Response[HTTPValidationError | SessionTemplate]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    template_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SessionTemplate]:
    r"""Update

     Update a session template's recipe fields. Omitted fields are preserved.

    The ``agent_version`` field uses sentinel-based partial-update semantics:
    omit it to preserve the current pin (or current \"track latest\" state),
    pass null to switch to \"track latest,\" pass a number to pin to that
    specific version. Already-spawned sessions are unaffected.

    Args:
        template_id (str):
        authorization (None | str | Unset):
        body (SessionTemplateUpdate): Request body for ``PUT /v1/session-templates/{id}``.

            Updates apply to future spawns only — already-spawned sessions are
            not retroactively migrated (see module docstring).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SessionTemplate]
    """

    kwargs = _get_kwargs(
        template_id=template_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    template_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SessionTemplate | None:
    r"""Update

     Update a session template's recipe fields. Omitted fields are preserved.

    The ``agent_version`` field uses sentinel-based partial-update semantics:
    omit it to preserve the current pin (or current \"track latest\" state),
    pass null to switch to \"track latest,\" pass a number to pin to that
    specific version. Already-spawned sessions are unaffected.

    Args:
        template_id (str):
        authorization (None | str | Unset):
        body (SessionTemplateUpdate): Request body for ``PUT /v1/session-templates/{id}``.

            Updates apply to future spawns only — already-spawned sessions are
            not retroactively migrated (see module docstring).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SessionTemplate
    """

    return sync_detailed(
        template_id=template_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    template_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SessionTemplate]:
    r"""Update

     Update a session template's recipe fields. Omitted fields are preserved.

    The ``agent_version`` field uses sentinel-based partial-update semantics:
    omit it to preserve the current pin (or current \"track latest\" state),
    pass null to switch to \"track latest,\" pass a number to pin to that
    specific version. Already-spawned sessions are unaffected.

    Args:
        template_id (str):
        authorization (None | str | Unset):
        body (SessionTemplateUpdate): Request body for ``PUT /v1/session-templates/{id}``.

            Updates apply to future spawns only — already-spawned sessions are
            not retroactively migrated (see module docstring).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SessionTemplate]
    """

    kwargs = _get_kwargs(
        template_id=template_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    template_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SessionTemplate | None:
    r"""Update

     Update a session template's recipe fields. Omitted fields are preserved.

    The ``agent_version`` field uses sentinel-based partial-update semantics:
    omit it to preserve the current pin (or current \"track latest\" state),
    pass null to switch to \"track latest,\" pass a number to pin to that
    specific version. Already-spawned sessions are unaffected.

    Args:
        template_id (str):
        authorization (None | str | Unset):
        body (SessionTemplateUpdate): Request body for ``PUT /v1/session-templates/{id}``.

            Updates apply to future spawns only — already-spawned sessions are
            not retroactively migrated (see module docstring).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SessionTemplate
    """

    return (
        await asyncio_detailed(
            template_id=template_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
