from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.testclient import TestClient

from aios.api.strict_query_params import reject_unknown_query_params


def _client() -> TestClient:
    app = FastAPI(dependencies=[Depends(reject_unknown_query_params)])

    @app.get("/items")
    async def items(
        direction: Annotated[str, Query(alias="dir")] = "forward",
        channel: Annotated[list[str] | None, Query()] = None,
    ) -> dict:
        return {"direction": direction, "channel": channel}

    @app.post("/v1/triggers/ingest/{token}")
    async def ingest(token: str, request: Request) -> dict:
        return {"token": token}

    return TestClient(app)


def test_unknown_query_parameter_is_rejected_and_named() -> None:
    response = _client().get("/items?seq=123")
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "seq"]


def test_aliases_and_repeatable_parameters_are_accepted() -> None:
    response = _client().get("/items?dir=backward&channel=a&channel=b")
    assert response.status_code == 200
    assert response.json() == {"direction": "backward", "channel": ["a", "b"]}


def test_raw_ingest_route_allows_provider_query_parameters() -> None:
    response = _client().post("/v1/triggers/ingest/secret?provider_retry=1")
    assert response.status_code == 200
