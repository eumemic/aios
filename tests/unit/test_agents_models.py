"""Pydantic validation for agent request models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.agents import AgentCreate, AgentUpdate


def _http_server(name: str, base_url: str) -> dict[str, str]:
    return {"name": name, "base_url": base_url}


class TestAgentCreateHttpServers:
    def test_rejects_duplicate_base_url_with_different_names(self) -> None:
        with pytest.raises(
            ValidationError, match=r"duplicate base_url 'https://api\.example\.com'"
        ):
            AgentCreate.model_validate(
                {
                    "name": "agent",
                    "model": "gpt-4",
                    "http_servers": [
                        _http_server("primary", "https://api.example.com"),
                        _http_server("secondary", "https://api.example.com"),
                    ],
                }
            )

    def test_accepts_duplicate_names_with_distinct_base_urls(self) -> None:
        agent = AgentCreate.model_validate(
            {
                "name": "agent",
                "model": "gpt-4",
                "http_servers": [
                    _http_server("api", "https://one.example.com"),
                    _http_server("api", "https://two.example.com"),
                ],
            }
        )

        assert [server.base_url for server in agent.http_servers] == [
            "https://one.example.com",
            "https://two.example.com",
        ]


class TestAgentUpdateHttpServers:
    def test_rejects_duplicate_base_url_with_different_names(self) -> None:
        with pytest.raises(
            ValidationError, match=r"duplicate base_url 'https://api\.example\.com'"
        ):
            AgentUpdate.model_validate(
                {
                    "version": 1,
                    "http_servers": [
                        _http_server("primary", "https://api.example.com"),
                        _http_server("secondary", "https://api.example.com"),
                    ],
                }
            )

    def test_accepts_duplicate_names_with_distinct_base_urls(self) -> None:
        update = AgentUpdate.model_validate(
            {
                "version": 1,
                "http_servers": [
                    _http_server("api", "https://one.example.com"),
                    _http_server("api", "https://two.example.com"),
                ],
            }
        )

        assert update.http_servers is not None
        assert [server.base_url for server in update.http_servers] == [
            "https://one.example.com",
            "https://two.example.com",
        ]

    def test_accepts_omitted_http_servers(self) -> None:
        update = AgentUpdate.model_validate({"version": 1})

        assert update.http_servers is None
