from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.gate_resume import GateResume
from ...models.http_validation_error import HTTPValidationError
from ...models.wf_run import WfRun
from ...types import UNSET, Response, Unset


def _get_kwargs(
    run_id: str,
    *,
    body: GateResume,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/runs/{run_id}/resume".format(
            run_id=quote(str(run_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WfRun | None:
    if response.status_code == 200:
        response_200 = WfRun.from_dict(response.json())

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
) -> Response[HTTPValidationError | WfRun]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GateResume,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRun]:
    """Resume Gate

     Resume a suspended gate by its ``gate_nonce``, delivering ``result``. Returns
    the updated run (its status flips back toward ``running``). A bad nonce or a
    cross-tenant run 404s.

    Args:
        run_id (str):
        authorization (None | str | Unset):
        body (GateResume): Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's
            value.

            Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
            ``call_started`` event), not the internal ``call_key``. ``result`` is the
            externally-delivered resume value (arbitrary JSON).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRun]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GateResume,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRun | None:
    """Resume Gate

     Resume a suspended gate by its ``gate_nonce``, delivering ``result``. Returns
    the updated run (its status flips back toward ``running``). A bad nonce or a
    cross-tenant run 404s.

    Args:
        run_id (str):
        authorization (None | str | Unset):
        body (GateResume): Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's
            value.

            Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
            ``call_started`` event), not the internal ``call_key``. ``result`` is the
            externally-delivered resume value (arbitrary JSON).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRun
    """

    return sync_detailed(
        run_id=run_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GateResume,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRun]:
    """Resume Gate

     Resume a suspended gate by its ``gate_nonce``, delivering ``result``. Returns
    the updated run (its status flips back toward ``running``). A bad nonce or a
    cross-tenant run 404s.

    Args:
        run_id (str):
        authorization (None | str | Unset):
        body (GateResume): Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's
            value.

            Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
            ``call_started`` event), not the internal ``call_key``. ``result`` is the
            externally-delivered resume value (arbitrary JSON).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRun]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: GateResume,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRun | None:
    """Resume Gate

     Resume a suspended gate by its ``gate_nonce``, delivering ``result``. Returns
    the updated run (its status flips back toward ``running``). A bad nonce or a
    cross-tenant run 404s.

    Args:
        run_id (str):
        authorization (None | str | Unset):
        body (GateResume): Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's
            value.

            Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
            ``call_started`` event), not the internal ``call_key``. ``result`` is the
            externally-delivered resume value (arbitrary JSON).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRun
    """

    return (
        await asyncio_detailed(
            run_id=run_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
