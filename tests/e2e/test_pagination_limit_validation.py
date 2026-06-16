"""E2E test: every list endpoint rejects out-of-range ``?limit=`` values.

Pre-fix: every paginated route declared ``limit: int = N`` with no
Pydantic constraints. Three symptoms:

* ``?limit=0`` returned ``has_more=True, data=[]`` (the standard
  ``has_more=len(items) == limit`` shortcut evaluated to ``0 == 0``
  → ``True``; clients that loop on ``has_more`` without null-checking
  ``next_after`` would spin).
* ``?limit=-1`` reached Postgres as ``LIMIT -1`` → ``asyncpg`` raised
  ``InvalidRowCountInLimitClauseError`` → unhandled 500.
* No upper bound — a single client request could pull unbounded rows
  (DoS surface; one request fetching ~1 GB of event data).

Fix: every paginated route now declares its ``limit`` via the shared
``PageLimit`` / ``EventPageLimit`` aliases, whose ceiling derives from
``MAX_PAGE_LIMIT`` / ``MAX_EVENT_PAGE_LIMIT``. Pydantic / FastAPI rejects
at the request boundary with 422; the routes never see invalid values.

Coverage is **route-derived, not hand-listed.** The earlier version of
this file enumerated 8 paths by hand and had already drifted — it omitted
``/v1/workflows`` (4 ``page_cursor`` sites) and every path-param'd list
endpoint, so those routes had *zero* limit-validation coverage. Instead of
re-curating that list, this test now walks ``app.routes`` for every handler
returning ``ListResponse[...]`` that declares a ``limit`` query param and
asserts each rejects ``limit=0``, ``limit=-1``, and ``limit=ceiling+1``.

The consequence (the actual foreclosure): a new list endpoint that forgets
the ``ge=1`` / ``le=…`` constraint is discovered here *by its return type*,
not by an author remembering to append it to a list — so the omission fails
CI whether or not the author knew the rule exists.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, NamedTuple

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


class _ListRoute(NamedTuple):
    """A discovered list endpoint and the data the test needs to probe it."""

    path: str  # concrete request path (path params filled with placeholders)
    ceiling: int  # the ``le=…`` cap declared on its ``limit`` query param

    @property
    def over_ceiling(self) -> int:
        return self.ceiling + 1


def _is_list_response(response_model: Any) -> bool:
    """True iff ``response_model`` is ``ListResponse`` or a parametrization of it.

    ``ListResponse[Foo]`` is a real pydantic generic subclass, so a plain
    ``issubclass`` catches the parametrized forms; the bare class is checked
    by identity. The fallback reads pydantic's generic metadata in case a
    future pydantic stops making the parametrization a subclass.
    """
    from aios.models.common import ListResponse

    if response_model is ListResponse:
        return True
    try:
        if isinstance(response_model, type) and issubclass(response_model, ListResponse):
            return True
    except TypeError:
        pass
    meta = getattr(response_model, "__pydantic_generic_metadata__", {})
    return meta.get("origin") is ListResponse


def _limit_ceiling(field_info: Any) -> int | None:
    """Extract the ``le=…`` ceiling from a FastAPI ``limit`` param, or None.

    FastAPI keeps the numeric constraints in ``field_info.metadata`` as
    annotated-types ``Le`` / ``Ge`` markers rather than plain attributes.
    A ``limit`` param with no ``le`` is treated as *uncapped* (None) — which
    the test flags, because an uncapped list endpoint is exactly the DoS
    regression this contract forecloses.
    """
    for marker in getattr(field_info, "metadata", []) or []:
        le = getattr(marker, "le", None)
        if le is not None:
            return int(le)
    le = getattr(field_info, "le", None)
    return int(le) if le is not None else None


def _discover_list_routes() -> list[_ListRoute]:
    """Walk the app's routes for every ``limit``-paginated list endpoint.

    Gated on (a) returning ``ListResponse[...]`` and (b) declaring a ``limit``
    query param — the gate avoids over-matching list endpoints that aren't
    cursor/limit paginated (e.g. ``/v1/runtime-tokens``). Path params are
    filled with a placeholder string: the request boundary validates the
    query ``limit`` (422) *before* the handler runs its resource lookup, so a
    non-existent id never shadows the validation we are asserting.
    """
    from fastapi.routing import APIRoute

    from aios.api.app import create_app

    routes: list[_ListRoute] = []
    for route in create_app().routes:
        if not isinstance(route, APIRoute):
            continue
        if "GET" not in route.methods:
            continue
        if not _is_list_response(route.response_model):
            continue
        limit_param = next((p for p in route.dependant.query_params if p.name == "limit"), None)
        if limit_param is None:
            continue
        ceiling = _limit_ceiling(limit_param.field_info)
        assert ceiling is not None, (
            f"{route.path} declares a paginated `limit` query param with no `le=` "
            f"ceiling — an unbounded list endpoint is a DoS surface. Declare it via "
            f"`PageLimit` / `EventPageLimit` (see aios.models.pagination)."
        )
        # Fill path params (``{store_id}`` → ``placeholder``) so the path is
        # requestable; the value never reaches a DB lookup before the 422.
        concrete = route.path
        for param in route.dependant.path_params:
            concrete = concrete.replace("{" + param.name + "}", "id_placeholder")
        routes.append(_ListRoute(path=concrete, ceiling=ceiling))
    return routes


# Discovered at import time so pytest can parametrize over it. If this is
# empty the introspection broke (renamed ListResponse, etc.) — fail loudly
# rather than silently testing nothing.
_LIST_ROUTES = _discover_list_routes()
assert _LIST_ROUTES, "route introspection found no `limit`-paginated list endpoints"


def _route_id(route: _ListRoute) -> str:
    return route.path


class TestPaginationLimitValidation:
    def test_introspection_covers_known_endpoints(self) -> None:
        """The walk must find the canonical path-free endpoints + the
        previously-drifted ones (``/v1/workflows`` and the path-param'd lists).

        Pins the discovery itself so a regression that makes the introspection
        match *nothing* (or drops the routes the old hand-list missed) is caught.
        """
        discovered = {r.path for r in _LIST_ROUTES}
        # The 8 the old hand-list covered, plus the ones it had drifted past.
        must_cover = {
            "/v1/agents",
            "/v1/sessions",
            "/v1/skills",
            "/v1/environments",
            "/v1/vaults",
            "/v1/connections",
            "/v1/memory-stores",
            "/v1/session-templates",
            "/v1/workflows",  # drifted: missing from the old hand-list
            "/v1/runs",
            "/v1/sessions/id_placeholder/events",  # path-param'd: uncovered before
            "/v1/runs/id_placeholder/events",
        }
        missing = must_cover - discovered
        assert not missing, f"route introspection no longer discovers: {sorted(missing)}"

    @pytest.mark.parametrize("route", _LIST_ROUTES, ids=_route_id)
    async def test_limit_zero_rejected_with_422(
        self, http_client: httpx.AsyncClient, route: _ListRoute
    ) -> None:
        r = await http_client.get(route.path, params={"limit": 0})
        assert r.status_code == 422, (
            f"{route.path}?limit=0 should reject with 422 (Pydantic Field ge=1); "
            f"got {r.status_code} body={r.text[:200]}"
        )

    @pytest.mark.parametrize("route", _LIST_ROUTES, ids=_route_id)
    async def test_limit_negative_rejected_with_422(
        self, http_client: httpx.AsyncClient, route: _ListRoute
    ) -> None:
        r = await http_client.get(route.path, params={"limit": -1})
        assert r.status_code == 422, (
            f"{route.path}?limit=-1 should reject with 422 (Pydantic Field ge=1); "
            f"got {r.status_code} body={r.text[:200]} — without the "
            f"constraint the value reaches Postgres as ``LIMIT -1`` and "
            f"raises asyncpg.InvalidRowCountInLimitClauseError → 500"
        )

    @pytest.mark.parametrize("route", _LIST_ROUTES, ids=_route_id)
    async def test_limit_over_max_rejected_with_422(
        self, http_client: httpx.AsyncClient, route: _ListRoute
    ) -> None:
        r = await http_client.get(route.path, params={"limit": route.over_ceiling})
        assert r.status_code == 422, (
            f"{route.path}?limit={route.over_ceiling} should reject with 422 "
            f"(Pydantic Field le={route.ceiling}); got {r.status_code} "
            f"body={r.text[:200]} — without the cap a single request can pull "
            f"unbounded rows (DoS surface)"
        )

    @pytest.mark.parametrize("route", _LIST_ROUTES, ids=_route_id)
    async def test_limit_at_ceiling_passes_validation(
        self, http_client: httpx.AsyncClient, route: _ListRoute
    ) -> None:
        """The ceiling value itself is ``le``-valid, so it must NOT 422.

        Confirms the cap is inclusive (``le``, not ``lt``) and that the only
        thing rejected is *out-of-range* — the endpoint may still 404 on a
        placeholder path id, but it must not fail request validation.
        """
        r = await http_client.get(route.path, params={"limit": route.ceiling})
        assert r.status_code != 422, (
            f"{route.path}?limit={route.ceiling} (== the ceiling) should pass "
            f"validation; got 422 body={r.text[:200]}"
        )
