from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_health_response_get_health import GetHealthResponseGetHealth
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/health",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> GetHealthResponseGetHealth | None:
    if response.status_code == 200:
        response_200 = GetHealthResponseGetHealth.from_dict(response.json())

        return response_200

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[GetHealthResponseGetHealth]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[GetHealthResponseGetHealth]:
    r"""Health

     Liveness probe. Unauthenticated; returns the running aios version.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>,
    \"source_commit\": <sha>}`` if the process is up. Deliberately does NOT
    touch the DB pool — a post-startup Postgres outage must not flip liveness
    (that's ``/ready``'s job), or an orchestrator would kill an otherwise-
    healthy process during a DB blip.

    ``source_commit`` echoes the container's build-time ``SOURCE_COMMIT`` env
    (baked into the image at ``docker build`` from Coolify's commit build-arg).
    It makes the *truthful running SHA* verifiable over pure HTTPS: ``fleet
    --drift`` only sees Coolify's deploy-queue commit (build INTENT), which can
    silently diverge from what a zombie/failed-deploy container is actually
    running — and reading the real SHA otherwise needs SSH, which is often
    blocked. Surfacing it here lets seat + ops audits confirm a promote landed
    and that api/worker are in lockstep over plain HTTP (issue #1669). Falls
    back to ``\"unknown\"`` outside a built container (local dev), so the key is
    always present for automated audits to key off of.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetHealthResponseGetHealth]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> GetHealthResponseGetHealth | None:
    r"""Health

     Liveness probe. Unauthenticated; returns the running aios version.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>,
    \"source_commit\": <sha>}`` if the process is up. Deliberately does NOT
    touch the DB pool — a post-startup Postgres outage must not flip liveness
    (that's ``/ready``'s job), or an orchestrator would kill an otherwise-
    healthy process during a DB blip.

    ``source_commit`` echoes the container's build-time ``SOURCE_COMMIT`` env
    (baked into the image at ``docker build`` from Coolify's commit build-arg).
    It makes the *truthful running SHA* verifiable over pure HTTPS: ``fleet
    --drift`` only sees Coolify's deploy-queue commit (build INTENT), which can
    silently diverge from what a zombie/failed-deploy container is actually
    running — and reading the real SHA otherwise needs SSH, which is often
    blocked. Surfacing it here lets seat + ops audits confirm a promote landed
    and that api/worker are in lockstep over plain HTTP (issue #1669). Falls
    back to ``\"unknown\"`` outside a built container (local dev), so the key is
    always present for automated audits to key off of.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetHealthResponseGetHealth
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[GetHealthResponseGetHealth]:
    r"""Health

     Liveness probe. Unauthenticated; returns the running aios version.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>,
    \"source_commit\": <sha>}`` if the process is up. Deliberately does NOT
    touch the DB pool — a post-startup Postgres outage must not flip liveness
    (that's ``/ready``'s job), or an orchestrator would kill an otherwise-
    healthy process during a DB blip.

    ``source_commit`` echoes the container's build-time ``SOURCE_COMMIT`` env
    (baked into the image at ``docker build`` from Coolify's commit build-arg).
    It makes the *truthful running SHA* verifiable over pure HTTPS: ``fleet
    --drift`` only sees Coolify's deploy-queue commit (build INTENT), which can
    silently diverge from what a zombie/failed-deploy container is actually
    running — and reading the real SHA otherwise needs SSH, which is often
    blocked. Surfacing it here lets seat + ops audits confirm a promote landed
    and that api/worker are in lockstep over plain HTTP (issue #1669). Falls
    back to ``\"unknown\"`` outside a built container (local dev), so the key is
    always present for automated audits to key off of.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetHealthResponseGetHealth]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> GetHealthResponseGetHealth | None:
    r"""Health

     Liveness probe. Unauthenticated; returns the running aios version.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>,
    \"source_commit\": <sha>}`` if the process is up. Deliberately does NOT
    touch the DB pool — a post-startup Postgres outage must not flip liveness
    (that's ``/ready``'s job), or an orchestrator would kill an otherwise-
    healthy process during a DB blip.

    ``source_commit`` echoes the container's build-time ``SOURCE_COMMIT`` env
    (baked into the image at ``docker build`` from Coolify's commit build-arg).
    It makes the *truthful running SHA* verifiable over pure HTTPS: ``fleet
    --drift`` only sees Coolify's deploy-queue commit (build INTENT), which can
    silently diverge from what a zombie/failed-deploy container is actually
    running — and reading the real SHA otherwise needs SSH, which is often
    blocked. Surfacing it here lets seat + ops audits confirm a promote landed
    and that api/worker are in lockstep over plain HTTP (issue #1669). Falls
    back to ``\"unknown\"`` outside a built container (local dev), so the key is
    always present for automated audits to key off of.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetHealthResponseGetHealth
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
