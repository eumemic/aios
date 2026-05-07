from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.skill_version import SkillVersion
from ...models.skill_version_create import SkillVersionCreate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    skill_id: str,
    *,
    body: SkillVersionCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/skills/{skill_id}/versions".format(
            skill_id=quote(str(skill_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SkillVersion | None:
    if response.status_code == 201:
        response_201 = SkillVersion.from_dict(response.json())

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
) -> Response[HTTPValidationError | SkillVersion]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    skill_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SkillVersionCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SkillVersion]:
    """Create Version

     Upload a new immutable version of a skill's file bundle.

    The bundle must include the same SKILL.md frontmatter shape as the
    initial create: directory, ``name``, ``description``. Each version is
    a complete snapshot — files not present in the upload are not carried
    over from previous versions.

    Args:
        skill_id (str):
        authorization (None | str | Unset):
        body (SkillVersionCreate): Request body for ``POST /v1/skills/{skill_id}/versions``.

            Same file format as :class:`SkillCreate`. The directory, name, and
            description are re-extracted from the new SKILL.md.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SkillVersion]
    """

    kwargs = _get_kwargs(
        skill_id=skill_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    skill_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SkillVersionCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SkillVersion | None:
    """Create Version

     Upload a new immutable version of a skill's file bundle.

    The bundle must include the same SKILL.md frontmatter shape as the
    initial create: directory, ``name``, ``description``. Each version is
    a complete snapshot — files not present in the upload are not carried
    over from previous versions.

    Args:
        skill_id (str):
        authorization (None | str | Unset):
        body (SkillVersionCreate): Request body for ``POST /v1/skills/{skill_id}/versions``.

            Same file format as :class:`SkillCreate`. The directory, name, and
            description are re-extracted from the new SKILL.md.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SkillVersion
    """

    return sync_detailed(
        skill_id=skill_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    skill_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SkillVersionCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SkillVersion]:
    """Create Version

     Upload a new immutable version of a skill's file bundle.

    The bundle must include the same SKILL.md frontmatter shape as the
    initial create: directory, ``name``, ``description``. Each version is
    a complete snapshot — files not present in the upload are not carried
    over from previous versions.

    Args:
        skill_id (str):
        authorization (None | str | Unset):
        body (SkillVersionCreate): Request body for ``POST /v1/skills/{skill_id}/versions``.

            Same file format as :class:`SkillCreate`. The directory, name, and
            description are re-extracted from the new SKILL.md.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SkillVersion]
    """

    kwargs = _get_kwargs(
        skill_id=skill_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    skill_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SkillVersionCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SkillVersion | None:
    """Create Version

     Upload a new immutable version of a skill's file bundle.

    The bundle must include the same SKILL.md frontmatter shape as the
    initial create: directory, ``name``, ``description``. Each version is
    a complete snapshot — files not present in the upload are not carried
    over from previous versions.

    Args:
        skill_id (str):
        authorization (None | str | Unset):
        body (SkillVersionCreate): Request body for ``POST /v1/skills/{skill_id}/versions``.

            Same file format as :class:`SkillCreate`. The directory, name, and
            description are re-extracted from the new SKILL.md.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SkillVersion
    """

    return (
        await asyncio_detailed(
            skill_id=skill_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
