from http import HTTPStatus
from io import BytesIO
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, File, Response, Unset


def _get_kwargs(
    session_id: str,
    file_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/files/{file_id}".format(
            session_id=quote(str(session_id), safe=""),
            file_id=quote(str(file_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> File | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = File(payload=BytesIO(response.content))

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
) -> Response[File | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    file_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[File | HTTPValidationError]:
    """Download File

     Stream back the bytes of a previously-uploaded file (#179).

    Operator-authenticated, same scoping as every other session-scoped
    read: 404s when the file doesn't exist, belongs to a different
    session, or isn't owned by the caller's account — a wrong session or a
    cross-account file id is indistinguishable from a missing file.  No
    transformation or resizing — this streams ``host_path`` verbatim.

    The *declared* content-type is not trusted on the way back out.
    ``stage_upload`` stores ``upload.content_type`` verbatim from the
    client's multipart header — no allowlist, no sniffing — so echoing it
    with ``inline`` would let an uploader pick the type their own bytes
    are rendered as.  ``image/svg+xml`` is the sharp edge: it passes any
    ``image/*`` prefix check and executes script in-origin.  Only types on
    :data:`INLINE_RENDERABLE_CONTENT_TYPES` (the raster images #179 needs)
    are served with their stored type inline; everything else degrades to
    ``application/octet-stream`` as an attachment.  ``nosniff`` covers the
    sniffing paths this allowlist doesn't enumerate.

    Args:
        session_id (str):
        file_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[File | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        file_id=file_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    file_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> File | HTTPValidationError | None:
    """Download File

     Stream back the bytes of a previously-uploaded file (#179).

    Operator-authenticated, same scoping as every other session-scoped
    read: 404s when the file doesn't exist, belongs to a different
    session, or isn't owned by the caller's account — a wrong session or a
    cross-account file id is indistinguishable from a missing file.  No
    transformation or resizing — this streams ``host_path`` verbatim.

    The *declared* content-type is not trusted on the way back out.
    ``stage_upload`` stores ``upload.content_type`` verbatim from the
    client's multipart header — no allowlist, no sniffing — so echoing it
    with ``inline`` would let an uploader pick the type their own bytes
    are rendered as.  ``image/svg+xml`` is the sharp edge: it passes any
    ``image/*`` prefix check and executes script in-origin.  Only types on
    :data:`INLINE_RENDERABLE_CONTENT_TYPES` (the raster images #179 needs)
    are served with their stored type inline; everything else degrades to
    ``application/octet-stream`` as an attachment.  ``nosniff`` covers the
    sniffing paths this allowlist doesn't enumerate.

    Args:
        session_id (str):
        file_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        File | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        file_id=file_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    file_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[File | HTTPValidationError]:
    """Download File

     Stream back the bytes of a previously-uploaded file (#179).

    Operator-authenticated, same scoping as every other session-scoped
    read: 404s when the file doesn't exist, belongs to a different
    session, or isn't owned by the caller's account — a wrong session or a
    cross-account file id is indistinguishable from a missing file.  No
    transformation or resizing — this streams ``host_path`` verbatim.

    The *declared* content-type is not trusted on the way back out.
    ``stage_upload`` stores ``upload.content_type`` verbatim from the
    client's multipart header — no allowlist, no sniffing — so echoing it
    with ``inline`` would let an uploader pick the type their own bytes
    are rendered as.  ``image/svg+xml`` is the sharp edge: it passes any
    ``image/*`` prefix check and executes script in-origin.  Only types on
    :data:`INLINE_RENDERABLE_CONTENT_TYPES` (the raster images #179 needs)
    are served with their stored type inline; everything else degrades to
    ``application/octet-stream`` as an attachment.  ``nosniff`` covers the
    sniffing paths this allowlist doesn't enumerate.

    Args:
        session_id (str):
        file_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[File | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        file_id=file_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    file_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> File | HTTPValidationError | None:
    """Download File

     Stream back the bytes of a previously-uploaded file (#179).

    Operator-authenticated, same scoping as every other session-scoped
    read: 404s when the file doesn't exist, belongs to a different
    session, or isn't owned by the caller's account — a wrong session or a
    cross-account file id is indistinguishable from a missing file.  No
    transformation or resizing — this streams ``host_path`` verbatim.

    The *declared* content-type is not trusted on the way back out.
    ``stage_upload`` stores ``upload.content_type`` verbatim from the
    client's multipart header — no allowlist, no sniffing — so echoing it
    with ``inline`` would let an uploader pick the type their own bytes
    are rendered as.  ``image/svg+xml`` is the sharp edge: it passes any
    ``image/*`` prefix check and executes script in-origin.  Only types on
    :data:`INLINE_RENDERABLE_CONTENT_TYPES` (the raster images #179 needs)
    are served with their stored type inline; everything else degrades to
    ``application/octet-stream`` as an attachment.  ``nosniff`` covers the
    sniffing paths this allowlist doesn't enumerate.

    Args:
        session_id (str):
        file_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        File | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            file_id=file_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
