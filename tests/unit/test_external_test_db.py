"""Tests for selecting an externally managed integration-test database."""

from __future__ import annotations

from typing import Any

import conftest


def test_db_url_uses_external_database_without_starting_docker(
    monkeypatch: Any, request: Any
) -> None:
    external_url = "postgresql://aios:aios@127.0.0.1:5432/aios_test"
    monkeypatch.setenv("AIOS_TEST_DB_URL", external_url)

    def fail_if_docker_is_probed() -> bool:
        raise AssertionError("external database selection must not probe Docker")

    monkeypatch.setattr(conftest, "_docker_available", fail_if_docker_is_probed)

    assert request.getfixturevalue("db_url") == external_url
