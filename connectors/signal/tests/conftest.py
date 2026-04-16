"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

BOT_UUID = "99999999-8888-7777-6666-555555555555"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES_DIR / name).read_text())
    return data


@pytest.fixture
def bot_uuid() -> str:
    return BOT_UUID


@pytest.fixture
def envelope_text_dm() -> dict[str, Any]:
    return _load("text_dm.json")


@pytest.fixture
def envelope_text_group() -> dict[str, Any]:
    return _load("text_group.json")


@pytest.fixture
def envelope_reaction() -> dict[str, Any]:
    return _load("reaction.json")


@pytest.fixture
def envelope_reply() -> dict[str, Any]:
    return _load("reply.json")


@pytest.fixture
def envelope_attachment_only() -> dict[str, Any]:
    return _load("attachment_only.json")


@pytest.fixture
def envelope_receipt() -> dict[str, Any]:
    return _load("receipt.json")


@pytest.fixture
def envelope_typing() -> dict[str, Any]:
    return _load("typing.json")


@pytest.fixture
def envelope_self() -> dict[str, Any]:
    return _load("self_message.json")
