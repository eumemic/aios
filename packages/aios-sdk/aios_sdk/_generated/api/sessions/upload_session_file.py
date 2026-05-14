from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_upload_session_file import BodyUploadSessionFile
from ...models.file_upload_response import FileUploadResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: BodyUploadSessionFile,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/files".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["files"] = body.to_multipart()

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> FileUploadResponse | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = FileUploadResponse.from_dict(response.json())

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
) -> Response[FileUploadResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: BodyUploadSessionFile,
    authorization: None | str | Unset = UNSET,
) -> Response[FileUploadResponse | HTTPValidationError]:
    """Upload File

     Upload a single file into the session's workspace (#324).

    Operator-authenticated.  Files land at a stable host path; the model
    sees them inside the sandbox at ``/mnt/uploads/<file_id>/<filename>``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (BodyUploadSessionFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[FileUploadResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: BodyUploadSessionFile,
    authorization: None | str | Unset = UNSET,
) -> FileUploadResponse | HTTPValidationError | None:
    """Upload File

     Upload a single file into the session's workspace (#324).

    Operator-authenticated.  Files land at a stable host path; the model
    sees them inside the sandbox at ``/mnt/uploads/<file_id>/<filename>``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (BodyUploadSessionFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        FileUploadResponse | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: BodyUploadSessionFile,
    authorization: None | str | Unset = UNSET,
) -> Response[FileUploadResponse | HTTPValidationError]:
    """Upload File

     Upload a single file into the session's workspace (#324).

    Operator-authenticated.  Files land at a stable host path; the model
    sees them inside the sandbox at ``/mnt/uploads/<file_id>/<filename>``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (BodyUploadSessionFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[FileUploadResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: BodyUploadSessionFile,
    authorization: None | str | Unset = UNSET,
) -> FileUploadResponse | HTTPValidationError | None:
    """Upload File

     Upload a single file into the session's workspace (#324).

    Operator-authenticated.  Files land at a stable host path; the model
    sees them inside the sandbox at ``/mnt/uploads/<file_id>/<filename>``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (BodyUploadSessionFile):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        FileUploadResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
