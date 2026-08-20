"""Tests for selecting an externally managed integration-test database."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def postgres_container() -> None:
    """Fail if ``db_url`` resolves its Docker-backed fallback fixture."""
    raise AssertionError("external database selection must not request Docker")


def test_db_url_uses_external_database_without_starting_docker(
    monkeypatch: Any, request: Any
) -> None:
    external_url = "postgresql://aios:aios@127.0.0.1:5432/aios_test"
    monkeypatch.setenv("AIOS_TEST_DB_URL", external_url)

    assert request.getfixturevalue("db_url") == external_url
