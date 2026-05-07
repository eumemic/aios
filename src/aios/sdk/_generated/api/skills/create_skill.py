from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.skill import Skill
from ...models.skill_create import SkillCreate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: SkillCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/skills",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Skill | None:
    if response.status_code == 201:
        response_201 = Skill.from_dict(response.json())

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
) -> Response[HTTPValidationError | Skill]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SkillCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Skill]:
    """Create

     Create a new skill from an uploaded file bundle.

    The bundle must include exactly one ``<directory>/SKILL.md`` file with
    YAML frontmatter containing at least a ``name`` field. The skill's
    ``directory``, ``name``, and ``description`` are extracted from that
    frontmatter; subsequent versions reuse this extraction.

    Args:
        authorization (None | str | Unset):
        body (SkillCreate): Request body for ``POST /v1/skills``.

            ``files`` must include exactly one ``{directory}/SKILL.md`` entry.
            The server extracts ``name``, ``description``, and ``directory`` from
            the SKILL.md frontmatter and file paths.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Skill]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: SkillCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Skill | None:
    """Create

     Create a new skill from an uploaded file bundle.

    The bundle must include exactly one ``<directory>/SKILL.md`` file with
    YAML frontmatter containing at least a ``name`` field. The skill's
    ``directory``, ``name``, and ``description`` are extracted from that
    frontmatter; subsequent versions reuse this extraction.

    Args:
        authorization (None | str | Unset):
        body (SkillCreate): Request body for ``POST /v1/skills``.

            ``files`` must include exactly one ``{directory}/SKILL.md`` entry.
            The server extracts ``name``, ``description``, and ``directory`` from
            the SKILL.md frontmatter and file paths.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Skill
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SkillCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Skill]:
    """Create

     Create a new skill from an uploaded file bundle.

    The bundle must include exactly one ``<directory>/SKILL.md`` file with
    YAML frontmatter containing at least a ``name`` field. The skill's
    ``directory``, ``name``, and ``description`` are extracted from that
    frontmatter; subsequent versions reuse this extraction.

    Args:
        authorization (None | str | Unset):
        body (SkillCreate): Request body for ``POST /v1/skills``.

            ``files`` must include exactly one ``{directory}/SKILL.md`` entry.
            The server extracts ``name``, ``description``, and ``directory`` from
            the SKILL.md frontmatter and file paths.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Skill]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SkillCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Skill | None:
    """Create

     Create a new skill from an uploaded file bundle.

    The bundle must include exactly one ``<directory>/SKILL.md`` file with
    YAML frontmatter containing at least a ``name`` field. The skill's
    ``directory``, ``name``, and ``description`` are extracted from that
    frontmatter; subsequent versions reuse this extraction.

    Args:
        authorization (None | str | Unset):
        body (SkillCreate): Request body for ``POST /v1/skills``.

            ``files`` must include exactly one ``{directory}/SKILL.md`` entry.
            The server extracts ``name``, ``description``, and ``directory`` from
            the SKILL.md frontmatter and file paths.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Skill
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
