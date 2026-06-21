"""Unit tests for the agent-acting skill builtins (skill_upsert / skill_archive).

These stub the worker pool + services so they need no live Postgres. They mirror
``tests/unit/test_workflow_management.py`` and cover the identity-load-bearing
invariants that don't depend on the DB:

* **F1** — a trusted id smuggled into the arguments is rejected by the tool schema
  (``extra="forbid"``) before the handler ever runs.
* **F2** — ``account_id`` derives from the executing session id (server-side), never
  from model input.
* **Error handling** — a service ``AiosError`` (bad ``SKILL.md``, unknown id)
  propagates unconverted, leaving the dispatch layer (``_classify_tool_error``) to
  decide eviction vs. a clean model-visible result. Only ``_parse`` bails locally
  as a ``ToolBail``.
* **Wiring** — ``transport == "agent_tool"`` and the union edit (``BuiltinToolType``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, get_args
from unittest.mock import ANY, AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.errors import ValidationError
from aios.models.agents import _BUILTIN_NAMES, BuiltinToolType
from aios.models.skills import Skill, SkillVersion
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import registry

_DT = datetime(2026, 1, 1, tzinfo=UTC)

_GOOD_FILES = {"foo/SKILL.md": "---\nname: foo\n---\nbody"}


def _skill(**over: Any) -> Skill:
    base: dict[str, Any] = dict(
        id="skl_1",
        display_title="Foo",
        latest_version=1,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Skill(**base)


def _sv(**over: Any) -> SkillVersion:
    base: dict[str, Any] = dict(
        skill_id="skl_1",
        version=1,
        directory="foo",
        name="foo",
        description="body",
        files={"SKILL.md": "---\nname: foo\n---\nbody"},
        created_at=_DT,
    )
    base.update(over)
    return SkillVersion(**base)


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    """Make ``require_pool`` + ``load_session_account_id`` cheap (no DB)."""
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value="acc_x")
    )


class TestSkillUpsert:
    async def test_skill_upsert_create(self, monkeypatch: Any) -> None:
        create = AsyncMock(return_value=(_skill(), _sv()))
        monkeypatch.setattr("aios.services.skills.create_skill", create)
        version_add = AsyncMock()
        monkeypatch.setattr("aios.services.skills.create_skill_version", version_add)

        result = await invoke_builtin(
            "ses_1",
            "skill_upsert",
            {"files": _GOOD_FILES, "display_title": "Foo"},
        )

        # account_id comes from the session stub, NOT from args (F2).
        create.assert_awaited_once_with(
            ANY,
            account_id="acc_x",
            display_title="Foo",
            files=_GOOD_FILES,
        )
        version_add.assert_not_awaited()
        assert result == {
            "skill_id": "skl_1",
            "version": 1,
            "name": "foo",
            "directory": "foo",
            "created": True,
        }

    async def test_skill_upsert_version(self, monkeypatch: Any) -> None:
        create = AsyncMock()
        monkeypatch.setattr("aios.services.skills.create_skill", create)
        version_add = AsyncMock(return_value=_sv(version=2))
        monkeypatch.setattr("aios.services.skills.create_skill_version", version_add)

        result = await invoke_builtin(
            "ses_1",
            "skill_upsert",
            {"skill_id": "skl_1", "files": _GOOD_FILES},
        )

        create.assert_not_awaited()
        version_add.assert_awaited_once_with(
            ANY,
            "skl_1",
            account_id="acc_x",
            files=_GOOD_FILES,
        )
        assert result == {
            "skill_id": "skl_1",
            "version": 2,
            "name": "foo",
            "directory": "foo",
            "created": False,
        }

    async def test_skill_upsert_create_requires_title(self, monkeypatch: Any) -> None:
        create = AsyncMock()
        monkeypatch.setattr("aios.services.skills.create_skill", create)
        with pytest.raises(ToolBail, match="display_title required"):
            await invoke_builtin("ses_1", "skill_upsert", {"files": _GOOD_FILES})
        create.assert_not_awaited()


class TestSchemaRejectsInjectedTrustedIds:
    """F1: the trusted ids are never schema fields, so injection is rejected up front."""

    async def test_injected_account_id_rejected(self, monkeypatch: Any) -> None:
        create = AsyncMock()
        monkeypatch.setattr("aios.services.skills.create_skill", create)
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "skill_upsert",
                {"files": _GOOD_FILES, "display_title": "Foo", "account_id": "acc_evil"},
            )
        create.assert_not_awaited()

    async def test_injected_session_id_rejected(self, monkeypatch: Any) -> None:
        create = AsyncMock()
        monkeypatch.setattr("aios.services.skills.create_skill", create)
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "skill_upsert",
                {"files": _GOOD_FILES, "display_title": "Foo", "session_id": "ses_victim"},
            )
        create.assert_not_awaited()

    async def test_injected_account_id_rejected_archive(self, monkeypatch: Any) -> None:
        archive = AsyncMock()
        monkeypatch.setattr("aios.services.skills.archive_skill", archive)
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "skill_archive",
                {"skill_id": "skl_1", "account_id": "acc_evil"},
            )
        archive.assert_not_awaited()


class TestSkillArchive:
    async def test_skill_archive(self, monkeypatch: Any) -> None:
        archive = AsyncMock(return_value=None)
        monkeypatch.setattr("aios.services.skills.archive_skill", archive)

        result = await invoke_builtin("ses_1", "skill_archive", {"skill_id": "skl_1"})

        archive.assert_awaited_once_with(ANY, "skl_1", account_id="acc_x")
        assert result == {"skill_id": "skl_1", "archived": True}


class TestErrorPropagation:
    async def test_service_error_propagates(self, monkeypatch: Any) -> None:
        # A bad SKILL.md surfaces from the service as a ValidationError (an AiosError),
        # which the handler lets propagate unconverted — NOT a ToolBail — so the dispatch
        # layer (_classify_tool_error) decides the model-visible result.
        monkeypatch.setattr(
            "aios.services.skills.create_skill",
            AsyncMock(side_effect=ValidationError("bad SKILL.md")),
        )
        with pytest.raises(ValidationError, match=r"bad SKILL\.md"):
            await invoke_builtin(
                "ses_1",
                "skill_upsert",
                {"files": {"foo/SKILL.md": "no frontmatter"}, "display_title": "Foo"},
            )


class TestRegistration:
    def test_registered_agent_tool_transport(self) -> None:
        assert registry.get("skill_upsert").transport == "agent_tool"
        assert registry.get("skill_archive").transport == "agent_tool"

    def test_builtin_names_includes_new_arms(self) -> None:
        assert "skill_upsert" in get_args(BuiltinToolType)
        assert "skill_archive" in get_args(BuiltinToolType)
        assert "skill_upsert" in _BUILTIN_NAMES
        assert "skill_archive" in _BUILTIN_NAMES
