"""Files uploaded into a session's workspace via ``POST /v1/sessions/<id>/files``."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class File(BaseModel):
    """Internal view of a row in the ``files`` table.

    ``host_path`` is service-layer-only and is stripped from the public
    response shape :class:`FileUploadResponse`.
    """

    id: str
    session_id: str
    filename: str
    host_path: str
    in_sandbox_path: str
    size: int
    content_type: str
    sha256: str
    created_at: datetime


class FileUploadResponse(BaseModel):
    """Wire shape returned from ``POST /v1/sessions/<id>/files``.

    ``file_id`` rather than ``id`` matches the #324 contract so callers can
    reference the file in future inbound attachment payloads without rebinding.
    """

    file_id: str
    in_sandbox_path: str
    filename: str
    size: int
    content_type: str
    sha256: str
