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

     Liveness probe. Unauthenticated; returns the running aios version + build SHA.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>, \"build_sha\": <sha-or-None>}``
    if the process is up. Deliberately does NOT touch the DB pool — a post-startup
    Postgres outage must not flip liveness (that's ``/ready``'s job), or an
    orchestrator would kill an otherwise-healthy process during a DB blip.

    ``build_sha`` (aios#1327, Unit A2) is the git commit baked into THIS image at
    build time (``ENV AIOS_BUILD_SHA`` in the Dockerfile ``base`` stage). It is the
    in-process running-truth read: unlike ``version`` (a static package string that
    never changes between builds), it tells you which commit is actually running.
    ``None`` when ``AIOS_BUILD_SHA`` is unset — an un-instrumented build — which the
    running==merged reconciler reads as ``cannot-determine``, never a false match.
    ``version`` stays the static package string, untouched.

    The return type is ``dict[str, str | None]`` (not ``dict[str, str]``): ``build_sha``
    is genuinely nullable because ``os.environ.get`` returns ``str | None``.

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

     Liveness probe. Unauthenticated; returns the running aios version + build SHA.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>, \"build_sha\": <sha-or-None>}``
    if the process is up. Deliberately does NOT touch the DB pool — a post-startup
    Postgres outage must not flip liveness (that's ``/ready``'s job), or an
    orchestrator would kill an otherwise-healthy process during a DB blip.

    ``build_sha`` (aios#1327, Unit A2) is the git commit baked into THIS image at
    build time (``ENV AIOS_BUILD_SHA`` in the Dockerfile ``base`` stage). It is the
    in-process running-truth read: unlike ``version`` (a static package string that
    never changes between builds), it tells you which commit is actually running.
    ``None`` when ``AIOS_BUILD_SHA`` is unset — an un-instrumented build — which the
    running==merged reconciler reads as ``cannot-determine``, never a false match.
    ``version`` stays the static package string, untouched.

    The return type is ``dict[str, str | None]`` (not ``dict[str, str]``): ``build_sha``
    is genuinely nullable because ``os.environ.get`` returns ``str | None``.

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

     Liveness probe. Unauthenticated; returns the running aios version + build SHA.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>, \"build_sha\": <sha-or-None>}``
    if the process is up. Deliberately does NOT touch the DB pool — a post-startup
    Postgres outage must not flip liveness (that's ``/ready``'s job), or an
    orchestrator would kill an otherwise-healthy process during a DB blip.

    ``build_sha`` (aios#1327, Unit A2) is the git commit baked into THIS image at
    build time (``ENV AIOS_BUILD_SHA`` in the Dockerfile ``base`` stage). It is the
    in-process running-truth read: unlike ``version`` (a static package string that
    never changes between builds), it tells you which commit is actually running.
    ``None`` when ``AIOS_BUILD_SHA`` is unset — an un-instrumented build — which the
    running==merged reconciler reads as ``cannot-determine``, never a false match.
    ``version`` stays the static package string, untouched.

    The return type is ``dict[str, str | None]`` (not ``dict[str, str]``): ``build_sha``
    is genuinely nullable because ``os.environ.get`` returns ``str | None``.

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

     Liveness probe. Unauthenticated; returns the running aios version + build SHA.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{\"status\": \"ok\", \"version\": <version>, \"build_sha\": <sha-or-None>}``
    if the process is up. Deliberately does NOT touch the DB pool — a post-startup
    Postgres outage must not flip liveness (that's ``/ready``'s job), or an
    orchestrator would kill an otherwise-healthy process during a DB blip.

    ``build_sha`` (aios#1327, Unit A2) is the git commit baked into THIS image at
    build time (``ENV AIOS_BUILD_SHA`` in the Dockerfile ``base`` stage). It is the
    in-process running-truth read: unlike ``version`` (a static package string that
    never changes between builds), it tells you which commit is actually running.
    ``None`` when ``AIOS_BUILD_SHA`` is unset — an un-instrumented build — which the
    running==merged reconciler reads as ``cannot-determine``, never a false match.
    ``version`` stays the static package string, untouched.

    The return type is ``dict[str, str | None]`` (not ``dict[str, str]``): ``build_sha``
    is genuinely nullable because ``os.environ.get`` returns ``str | None``.

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
