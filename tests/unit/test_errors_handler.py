"""Unit tests for ``aios.errors.validation_error_handler``.

Pydantic's ``value_error`` entries include a ``ctx.error`` field that is a
live ``ValueError`` instance — unserializable by ``json.dumps``. Before
the fix the handler itself raised ``TypeError`` and Starlette fell back
to a generic 500.
"""

from __future__ import annotations

import io
import json

from fastapi import FastAPI, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from starlette.requests import Request

from aios.errors import install_exception_handlers, validation_error_handler


class _CustomModel(BaseModel):
    x: str

    @field_validator("x")
    @classmethod
    def _v(cls, v: str) -> str:
        if v == "bad":
            raise ValueError("x must not be 'bad'")
        return v


def test_field_validator_value_error_custom_model() -> None:
    """Minimal synthetic model proves the handler strips ``ctx`` for any
    ``@field_validator`` that raises ``ValueError``."""
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/echo")
    async def _echo(body: _CustomModel) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)
    response = client.post("/echo", json={"x": "bad"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "validation_error"
    errors = body["error"]["detail"]["errors"]
    assert all("ctx" not in err for err in errors)
    assert any("x must not be 'bad'" in err.get("msg", "") for err in errors)


class _IntModel(BaseModel):
    n: int


def test_regular_type_error_still_422() -> None:
    """Plain type-mismatch validation (no custom validator, no ctx) still
    renders as 422."""
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/echo")
    async def _echo(body: _IntModel) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)
    response = client.post("/echo", json={"n": "not-an-int"})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


async def test_non_serializable_input_does_not_crash_handler() -> None:
    """A multipart validation error carries the offending ``UploadFile`` in its
    ``input`` field, which ``json.dumps`` can't serialize. Before the fix the
    handler's own ``JSONResponse`` raised ``TypeError`` and Starlette fell back
    to a bare 500 — losing the 422 and the error envelope. The error list must
    be coerced JSON-safe, not just have ``ctx`` stripped."""
    upload = UploadFile(file=io.BytesIO(b"bytes"), filename="photo.png")
    exc = RequestValidationError(
        errors=[{"type": "model_attributes_type", "loc": ("body",), "msg": "bad", "input": upload}]
    )
    request = Request(
        {"type": "http", "method": "POST", "path": "/upload", "headers": [], "query_string": b""}
    )

    response = await validation_error_handler(request, exc)

    assert response.status_code == 422
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "validation_error"
    # The non-serializable ``input`` is coerced to a string, not crashed on.
    assert isinstance(body["error"]["detail"]["errors"][0]["input"], str)
