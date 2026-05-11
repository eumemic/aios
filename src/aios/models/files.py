"""Files uploaded into a session's workspace via ``POST /v1/sessions/<id>/files``.

A file is a session-scoped, immutable-from-the-model's-POV blob staged into
the api's filesystem and exposed read-only to the sandbox at
``/mnt/uploads/<file_id>/<filename>``.  Connectors upload bytes through the
api instead of placing them on a shared mount; the model accesses uploads
through the same standard tools (``read``, ``bash``, …) it uses for any
other path under a known bind.

``host_path`` is the internal location on the api's filesystem and is NOT
exposed in API responses — see :class:`FileUploadResponse` for the wire shape.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class File(BaseModel):
    """Internal view of a row in the ``files`` table.

    Includes ``host_path`` so service-layer code can locate the bytes; the
    public response shape strips it.
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

    The ``file_id`` name (rather than the internal ``id``) matches the
    contract in #324 so callers can reference the file in future inbound
    attachment payloads without rebinding.
    """

    file_id: str = Field(description="Stable id of the uploaded file.")
    in_sandbox_path: str = Field(
        description="Path inside the sandbox container where the model sees the file."
    )
    filename: str
    size: int
    content_type: str
    sha256: str
