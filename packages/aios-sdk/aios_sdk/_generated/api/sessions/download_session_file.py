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
    client's multipart header — no allowlist, no sniffing — so both the
    declared type AND the bytes are attacker-chosen, independently.  That
    is two distinct attack shapes, and **each is closed by a different
    control here.  Neither control covers both; removing either reopens
    one of them.**

    1. *Dangerous declared type, any bytes.*  ``image/svg+xml`` passes an
       ``image/*`` prefix test and executes script in the serving origin.
       Closed by :data:`INLINE_RENDERABLE_CONTENT_TYPES`: only exact
       members are served inline under their stored type, and everything
       else is re-typed to ``application/octet-stream`` and served as an
       attachment.  ``nosniff`` does nothing against this — the declared
       type is honoured, not sniffed, and the attack lands anyway.

    2. *Safe declared type, dangerous bytes.*  HTML with a ``<script>``
       uploaded as ``image/png``.  That type IS allowlisted, so this is
       served inline as ``image/png`` — by design, and correctly, because
       the endpoint cannot afford to inspect bytes.  What stops it is
       ``X-Content-Type-Options: nosniff``: without it a browser may sniff
       HTML out of a response labelled ``image/png`` and render it as a
       document in this origin.  With it the label is binding and the
       response is an inert broken image.  **The allowlist does nothing
       against this — the declared type is on it.**

    So ``nosniff`` is not belt-and-braces; for shape 2 it is the only
    control, which is why it is set unconditionally on both branches.
    ``test_inline_allowlist_serves_untrusted_bytes_with_nosniff`` is the
    regression guard: it is the test that fails if the header is dropped
    during a cleanup, since every other test passes without it.

    Scoping is unchanged from every other session-scoped read: 404s when
    the file doesn't exist, belongs to a different session, or isn't owned
    by the caller's account — a wrong session or a cross-account file id is
    indistinguishable from a missing file.  No transformation or resizing;
    ``host_path`` is streamed verbatim.  This is a rendering control, not a
    filter: bytes are returned byte-identical on both branches.

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
    client's multipart header — no allowlist, no sniffing — so both the
    declared type AND the bytes are attacker-chosen, independently.  That
    is two distinct attack shapes, and **each is closed by a different
    control here.  Neither control covers both; removing either reopens
    one of them.**

    1. *Dangerous declared type, any bytes.*  ``image/svg+xml`` passes an
       ``image/*`` prefix test and executes script in the serving origin.
       Closed by :data:`INLINE_RENDERABLE_CONTENT_TYPES`: only exact
       members are served inline under their stored type, and everything
       else is re-typed to ``application/octet-stream`` and served as an
       attachment.  ``nosniff`` does nothing against this — the declared
       type is honoured, not sniffed, and the attack lands anyway.

    2. *Safe declared type, dangerous bytes.*  HTML with a ``<script>``
       uploaded as ``image/png``.  That type IS allowlisted, so this is
       served inline as ``image/png`` — by design, and correctly, because
       the endpoint cannot afford to inspect bytes.  What stops it is
       ``X-Content-Type-Options: nosniff``: without it a browser may sniff
       HTML out of a response labelled ``image/png`` and render it as a
       document in this origin.  With it the label is binding and the
       response is an inert broken image.  **The allowlist does nothing
       against this — the declared type is on it.**

    So ``nosniff`` is not belt-and-braces; for shape 2 it is the only
    control, which is why it is set unconditionally on both branches.
    ``test_inline_allowlist_serves_untrusted_bytes_with_nosniff`` is the
    regression guard: it is the test that fails if the header is dropped
    during a cleanup, since every other test passes without it.

    Scoping is unchanged from every other session-scoped read: 404s when
    the file doesn't exist, belongs to a different session, or isn't owned
    by the caller's account — a wrong session or a cross-account file id is
    indistinguishable from a missing file.  No transformation or resizing;
    ``host_path`` is streamed verbatim.  This is a rendering control, not a
    filter: bytes are returned byte-identical on both branches.

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
    client's multipart header — no allowlist, no sniffing — so both the
    declared type AND the bytes are attacker-chosen, independently.  That
    is two distinct attack shapes, and **each is closed by a different
    control here.  Neither control covers both; removing either reopens
    one of them.**

    1. *Dangerous declared type, any bytes.*  ``image/svg+xml`` passes an
       ``image/*`` prefix test and executes script in the serving origin.
       Closed by :data:`INLINE_RENDERABLE_CONTENT_TYPES`: only exact
       members are served inline under their stored type, and everything
       else is re-typed to ``application/octet-stream`` and served as an
       attachment.  ``nosniff`` does nothing against this — the declared
       type is honoured, not sniffed, and the attack lands anyway.

    2. *Safe declared type, dangerous bytes.*  HTML with a ``<script>``
       uploaded as ``image/png``.  That type IS allowlisted, so this is
       served inline as ``image/png`` — by design, and correctly, because
       the endpoint cannot afford to inspect bytes.  What stops it is
       ``X-Content-Type-Options: nosniff``: without it a browser may sniff
       HTML out of a response labelled ``image/png`` and render it as a
       document in this origin.  With it the label is binding and the
       response is an inert broken image.  **The allowlist does nothing
       against this — the declared type is on it.**

    So ``nosniff`` is not belt-and-braces; for shape 2 it is the only
    control, which is why it is set unconditionally on both branches.
    ``test_inline_allowlist_serves_untrusted_bytes_with_nosniff`` is the
    regression guard: it is the test that fails if the header is dropped
    during a cleanup, since every other test passes without it.

    Scoping is unchanged from every other session-scoped read: 404s when
    the file doesn't exist, belongs to a different session, or isn't owned
    by the caller's account — a wrong session or a cross-account file id is
    indistinguishable from a missing file.  No transformation or resizing;
    ``host_path`` is streamed verbatim.  This is a rendering control, not a
    filter: bytes are returned byte-identical on both branches.

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
    client's multipart header — no allowlist, no sniffing — so both the
    declared type AND the bytes are attacker-chosen, independently.  That
    is two distinct attack shapes, and **each is closed by a different
    control here.  Neither control covers both; removing either reopens
    one of them.**

    1. *Dangerous declared type, any bytes.*  ``image/svg+xml`` passes an
       ``image/*`` prefix test and executes script in the serving origin.
       Closed by :data:`INLINE_RENDERABLE_CONTENT_TYPES`: only exact
       members are served inline under their stored type, and everything
       else is re-typed to ``application/octet-stream`` and served as an
       attachment.  ``nosniff`` does nothing against this — the declared
       type is honoured, not sniffed, and the attack lands anyway.

    2. *Safe declared type, dangerous bytes.*  HTML with a ``<script>``
       uploaded as ``image/png``.  That type IS allowlisted, so this is
       served inline as ``image/png`` — by design, and correctly, because
       the endpoint cannot afford to inspect bytes.  What stops it is
       ``X-Content-Type-Options: nosniff``: without it a browser may sniff
       HTML out of a response labelled ``image/png`` and render it as a
       document in this origin.  With it the label is binding and the
       response is an inert broken image.  **The allowlist does nothing
       against this — the declared type is on it.**

    So ``nosniff`` is not belt-and-braces; for shape 2 it is the only
    control, which is why it is set unconditionally on both branches.
    ``test_inline_allowlist_serves_untrusted_bytes_with_nosniff`` is the
    regression guard: it is the test that fails if the header is dropped
    during a cleanup, since every other test passes without it.

    Scoping is unchanged from every other session-scoped read: 404s when
    the file doesn't exist, belongs to a different session, or isn't owned
    by the caller's account — a wrong session or a cross-account file id is
    indistinguishable from a missing file.  No transformation or resizing;
    ``host_path`` is streamed verbatim.  This is a rendering control, not a
    filter: bytes are returned byte-identical on both branches.

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
