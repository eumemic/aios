"""Unit tests for ``aios.errors.validation_error_handler``.

Pydantic's ``value_error`` entries include a ``ctx.error`` field that is a
live ``ValueError`` instance — unserializable by ``json.dumps``. Before
the fix the handler itself raised ``TypeError`` and Starlette fell back
to a generic 500. The routing-rule prefix validator in
``aios.models.routing_rules.RoutingRuleCreate`` is a real trigger: a
``prefix`` like ``"a/"`` fails ``validate_path_segments`` with a raised
``ValueError``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator

from aios.errors import install_exception_handlers
from aios.models.routing_rules import RoutingRuleCreate


def test_field_validator_value_error_returns_422_not_500() -> None:
    """A body that trips a ``@field_validator`` raising ``ValueError``
    must render as 422 in our envelope, not Starlette's generic 500.

    Uses the real ``RoutingRuleCreate`` with the problematic prefix
    from issue #121: ``"a/"`` (trailing empty segment).
    """
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/echo")
    async def _echo(body: RoutingRuleCreate) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)
    response = client.post("/echo", json={"prefix": "a/", "target": "session:s_x"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["message"] == "request body failed validation"
    errors = body["error"]["detail"]["errors"]
    assert isinstance(errors, list) and len(errors) >= 1
    # ctx must be stripped — it's what held the unserializable ValueError.
    for err in errors:
        assert "ctx" not in err
    # The prefix validator's message should be visible in msg.
    assert any("must not contain empty segments" in err.get("msg", "") for err in errors)


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
