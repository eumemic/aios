"""Conftest for lanes unit tests.

The parent ``tests/unit/conftest.py`` declares several ``autouse`` fixtures
that patch deep into ``aios.services.sessions``, ``aios.harness.runtime``,
etc. Those modules transitively import ``aios_connector_http`` and other
heavy dependencies that are not installed in every sandbox.

The tests in this sub-package are **pure-data / structural** tests against
``aios.lanes.models`` and the script text constant — they never touch
services, harness, or DB layers.  We override each problematic autouse
fixture with a trivial no-op so collection succeeds without the heavy
import chain.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _unit_runtime_tool_provider() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _unit_runtime_tool_broker() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _unit_load_session_account_id_stub() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _unit_no_session_cancel_harvest() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _unit_no_scan_floor_advance() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _unit_spend_state_ungated() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _unit_provider_auth_ungated() -> Iterator[None]:
    yield
