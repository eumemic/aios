from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_usage_consumers_metric import ListUsageConsumersMetric
from ...models.usage_consumers_response import UsageConsumersResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    window_seconds: int | Unset = 86400,
    metric: ListUsageConsumersMetric | Unset = ListUsageConsumersMetric.COST_MICROUSD,
    limit: int | Unset = 20,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["window_seconds"] = window_seconds

    json_metric: str | Unset = UNSET
    if not isinstance(metric, Unset):
        json_metric = metric.value

    params["metric"] = json_metric

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/usage/consumers",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | UsageConsumersResponse | None:
    if response.status_code == 200:
        response_200 = UsageConsumersResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | UsageConsumersResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    window_seconds: int | Unset = 86400,
    metric: ListUsageConsumersMetric | Unset = ListUsageConsumersMetric.COST_MICROUSD,
    limit: int | Unset = 20,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | UsageConsumersResponse]:
    """List Usage Consumers

     Rank root consumers by rolling creation-subtree inference rate.

    Root consumers are additive: every session/run belongs to exactly one root
    through immutable creation edges, so ``share`` values never double-count
    shared invocation work. Archived descendants remain in the rollup. Rates
    update on every inference charge and are normalized per hour over the
    requested rolling window.

    Args:
        window_seconds (int | Unset):  Default: 86400.
        metric (ListUsageConsumersMetric | Unset):  Default:
            ListUsageConsumersMetric.COST_MICROUSD.
        limit (int | Unset):  Default: 20.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | UsageConsumersResponse]
    """

    kwargs = _get_kwargs(
        window_seconds=window_seconds,
        metric=metric,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    window_seconds: int | Unset = 86400,
    metric: ListUsageConsumersMetric | Unset = ListUsageConsumersMetric.COST_MICROUSD,
    limit: int | Unset = 20,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | UsageConsumersResponse | None:
    """List Usage Consumers

     Rank root consumers by rolling creation-subtree inference rate.

    Root consumers are additive: every session/run belongs to exactly one root
    through immutable creation edges, so ``share`` values never double-count
    shared invocation work. Archived descendants remain in the rollup. Rates
    update on every inference charge and are normalized per hour over the
    requested rolling window.

    Args:
        window_seconds (int | Unset):  Default: 86400.
        metric (ListUsageConsumersMetric | Unset):  Default:
            ListUsageConsumersMetric.COST_MICROUSD.
        limit (int | Unset):  Default: 20.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | UsageConsumersResponse
    """

    return sync_detailed(
        client=client,
        window_seconds=window_seconds,
        metric=metric,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    window_seconds: int | Unset = 86400,
    metric: ListUsageConsumersMetric | Unset = ListUsageConsumersMetric.COST_MICROUSD,
    limit: int | Unset = 20,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | UsageConsumersResponse]:
    """List Usage Consumers

     Rank root consumers by rolling creation-subtree inference rate.

    Root consumers are additive: every session/run belongs to exactly one root
    through immutable creation edges, so ``share`` values never double-count
    shared invocation work. Archived descendants remain in the rollup. Rates
    update on every inference charge and are normalized per hour over the
    requested rolling window.

    Args:
        window_seconds (int | Unset):  Default: 86400.
        metric (ListUsageConsumersMetric | Unset):  Default:
            ListUsageConsumersMetric.COST_MICROUSD.
        limit (int | Unset):  Default: 20.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | UsageConsumersResponse]
    """

    kwargs = _get_kwargs(
        window_seconds=window_seconds,
        metric=metric,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    window_seconds: int | Unset = 86400,
    metric: ListUsageConsumersMetric | Unset = ListUsageConsumersMetric.COST_MICROUSD,
    limit: int | Unset = 20,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | UsageConsumersResponse | None:
    """List Usage Consumers

     Rank root consumers by rolling creation-subtree inference rate.

    Root consumers are additive: every session/run belongs to exactly one root
    through immutable creation edges, so ``share`` values never double-count
    shared invocation work. Archived descendants remain in the rollup. Rates
    update on every inference charge and are normalized per hour over the
    requested rolling window.

    Args:
        window_seconds (int | Unset):  Default: 86400.
        metric (ListUsageConsumersMetric | Unset):  Default:
            ListUsageConsumersMetric.COST_MICROUSD.
        limit (int | Unset):  Default: 20.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | UsageConsumersResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            window_seconds=window_seconds,
            metric=metric,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
