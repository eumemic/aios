from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.bootstrap_request import BootstrapRequest
from ...models.bootstrap_response import BootstrapResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/accounts/bootstrap",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> BootstrapResponse | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = BootstrapResponse.from_dict(response.json())

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
) -> Response[BootstrapResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[BootstrapResponse | HTTPValidationError]:
    r"""Bootstrap

     Mint the root account and its first API key.

    T2 decision (#1463): ``bootstrap`` is deliberately retained over
    ``create_root_account`` — the one-time, token-gated, self-closing
    operator ceremony semantic is load-bearing (the endpoint behaves like it
    doesn't exist once a root is in place), and a generic ``create`` would
    obscure that. Documented exception; not part of the agent-facing plane.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value.

    Root-exists check fires before the token check so a probe with no/
    wrong token can't distinguish \"no bootstrap\" (404) from \"wrong token
    but bootstrap is still open\" (401). Once a root is in place the
    endpoint behaves like it doesn't exist at all.

    ``plaintext_key`` in the response is the only time the operator key
    is returned in plaintext.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BootstrapResponse | HTTPValidationError]
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
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> BootstrapResponse | HTTPValidationError | None:
    r"""Bootstrap

     Mint the root account and its first API key.

    T2 decision (#1463): ``bootstrap`` is deliberately retained over
    ``create_root_account`` — the one-time, token-gated, self-closing
    operator ceremony semantic is load-bearing (the endpoint behaves like it
    doesn't exist once a root is in place), and a generic ``create`` would
    obscure that. Documented exception; not part of the agent-facing plane.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value.

    Root-exists check fires before the token check so a probe with no/
    wrong token can't distinguish \"no bootstrap\" (404) from \"wrong token
    but bootstrap is still open\" (401). Once a root is in place the
    endpoint behaves like it doesn't exist at all.

    ``plaintext_key`` in the response is the only time the operator key
    is returned in plaintext.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BootstrapResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[BootstrapResponse | HTTPValidationError]:
    r"""Bootstrap

     Mint the root account and its first API key.

    T2 decision (#1463): ``bootstrap`` is deliberately retained over
    ``create_root_account`` — the one-time, token-gated, self-closing
    operator ceremony semantic is load-bearing (the endpoint behaves like it
    doesn't exist once a root is in place), and a generic ``create`` would
    obscure that. Documented exception; not part of the agent-facing plane.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value.

    Root-exists check fires before the token check so a probe with no/
    wrong token can't distinguish \"no bootstrap\" (404) from \"wrong token
    but bootstrap is still open\" (401). Once a root is in place the
    endpoint behaves like it doesn't exist at all.

    ``plaintext_key`` in the response is the only time the operator key
    is returned in plaintext.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BootstrapResponse | HTTPValidationError]
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
    body: BootstrapRequest,
    authorization: None | str | Unset = UNSET,
) -> BootstrapResponse | HTTPValidationError | None:
    r"""Bootstrap

     Mint the root account and its first API key.

    T2 decision (#1463): ``bootstrap`` is deliberately retained over
    ``create_root_account`` — the one-time, token-gated, self-closing
    operator ceremony semantic is load-bearing (the endpoint behaves like it
    doesn't exist once a root is in place), and a generic ``create`` would
    obscure that. Documented exception; not part of the agent-facing plane.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value.

    Root-exists check fires before the token check so a probe with no/
    wrong token can't distinguish \"no bootstrap\" (404) from \"wrong token
    but bootstrap is still open\" (401). Once a root is in place the
    endpoint behaves like it doesn't exist at all.

    ``plaintext_key`` in the response is the only time the operator key
    is returned in plaintext.

    Args:
        authorization (None | str | Unset):
        body (BootstrapRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BootstrapResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
