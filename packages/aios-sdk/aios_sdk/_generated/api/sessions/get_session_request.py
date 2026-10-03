from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.missing_request import MissingRequest
from ...models.rebuilt_request import RebuiltRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    request_id: str,
    *,
    model: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_model: None | str | Unset
    if isinstance(model, Unset):
        json_model = UNSET
    else:
        json_model = model
    params["model"] = json_model

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/requests/{request_id}".format(
            session_id=quote(str(session_id), safe=""),
            request_id=quote(str(request_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | MissingRequest | RebuiltRequest | None:
    if response.status_code == 200:

        def _parse_response_200(data: object) -> MissingRequest | RebuiltRequest:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_0 = RebuiltRequest.from_dict(data)

                return response_200_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_200_type_1 = MissingRequest.from_dict(data)

            return response_200_type_1

        response_200 = _parse_response_200(response.json())

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
) -> Response[HTTPValidationError | MissingRequest | RebuiltRequest]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    request_id: str,
    *,
    client: AuthenticatedClient | Client,
    model: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MissingRequest | RebuiltRequest]:
    """Get Request

     Rebuild a request the session sent (#2471).

    ``request_id`` is the id of the span that opened the send: a
    ``model_request_start`` event, or a ``model_workflow_park`` event for a
    workflow-bound agent. The request is recomposed from its captured record,
    blobs and the event log, using today's renderer, and reported with its
    fidelity. ``?model=`` renders it for another model's vision and thinking gates
    instead. 404 when the event isn't a captured request.

    Attachment files are read where the API runs: a request that inlined an image
    this process can't read reports ``missing: attachment``.

    Args:
        session_id (str):
        request_id (str):
        model (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MissingRequest | RebuiltRequest]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        request_id=request_id,
        model=model,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    request_id: str,
    *,
    client: AuthenticatedClient | Client,
    model: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MissingRequest | RebuiltRequest | None:
    """Get Request

     Rebuild a request the session sent (#2471).

    ``request_id`` is the id of the span that opened the send: a
    ``model_request_start`` event, or a ``model_workflow_park`` event for a
    workflow-bound agent. The request is recomposed from its captured record,
    blobs and the event log, using today's renderer, and reported with its
    fidelity. ``?model=`` renders it for another model's vision and thinking gates
    instead. 404 when the event isn't a captured request.

    Attachment files are read where the API runs: a request that inlined an image
    this process can't read reports ``missing: attachment``.

    Args:
        session_id (str):
        request_id (str):
        model (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MissingRequest | RebuiltRequest
    """

    return sync_detailed(
        session_id=session_id,
        request_id=request_id,
        client=client,
        model=model,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    request_id: str,
    *,
    client: AuthenticatedClient | Client,
    model: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MissingRequest | RebuiltRequest]:
    """Get Request

     Rebuild a request the session sent (#2471).

    ``request_id`` is the id of the span that opened the send: a
    ``model_request_start`` event, or a ``model_workflow_park`` event for a
    workflow-bound agent. The request is recomposed from its captured record,
    blobs and the event log, using today's renderer, and reported with its
    fidelity. ``?model=`` renders it for another model's vision and thinking gates
    instead. 404 when the event isn't a captured request.

    Attachment files are read where the API runs: a request that inlined an image
    this process can't read reports ``missing: attachment``.

    Args:
        session_id (str):
        request_id (str):
        model (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MissingRequest | RebuiltRequest]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        request_id=request_id,
        model=model,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    request_id: str,
    *,
    client: AuthenticatedClient | Client,
    model: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MissingRequest | RebuiltRequest | None:
    """Get Request

     Rebuild a request the session sent (#2471).

    ``request_id`` is the id of the span that opened the send: a
    ``model_request_start`` event, or a ``model_workflow_park`` event for a
    workflow-bound agent. The request is recomposed from its captured record,
    blobs and the event log, using today's renderer, and reported with its
    fidelity. ``?model=`` renders it for another model's vision and thinking gates
    instead. 404 when the event isn't a captured request.

    Attachment files are read where the API runs: a request that inlined an image
    this process can't read reports ``missing: attachment``.

    Args:
        session_id (str):
        request_id (str):
        model (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MissingRequest | RebuiltRequest
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            request_id=request_id,
            client=client,
            model=model,
            authorization=authorization,
        )
    ).parsed
