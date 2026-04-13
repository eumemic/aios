"""E2E tests for the skills system.

Tests run against a real testcontainer Postgres with migrations applied.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.models.skills import AgentSkillRef


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


_SKILL_MD = (
    "---\nname: test-skill\ndescription: A test skill for e2e\n---\n# Instructions\nDo stuff."
)
_FILES = {"test-skill/SKILL.md": _SKILL_MD}


class TestSkillCRUD:
    async def test_create_and_get(self, pool: Any) -> None:
        from aios.services import skills as svc

        skill, version = await svc.create_skill(pool, display_title="Test Skill", files=_FILES)
        assert skill.id.startswith("skl_")
        assert skill.display_title == "Test Skill"
        assert skill.latest_version == 1
        assert skill.archived_at is None

        assert version.skill_id == skill.id
        assert version.version == 1
        assert version.directory == "test-skill"
        assert version.name == "test-skill"
        assert version.description == "A test skill for e2e"
        assert "SKILL.md" in version.files

        fetched = await svc.get_skill(pool, skill.id)
        assert fetched.id == skill.id

    async def test_list_skills(self, pool: Any) -> None:
        from aios.services import skills as svc

        s1, _ = await svc.create_skill(
            pool,
            display_title="List-A",
            files={"list-a/SKILL.md": "---\nname: list-a\n---\n# A"},
        )
        s2, _ = await svc.create_skill(
            pool,
            display_title="List-B",
            files={"list-b/SKILL.md": "---\nname: list-b\n---\n# B"},
        )
        skills = await svc.list_skills(pool, limit=100)
        ids = [s.id for s in skills]
        assert s1.id in ids
        assert s2.id in ids

    async def test_archive_skill(self, pool: Any) -> None:
        from aios.services import skills as svc

        skill, _ = await svc.create_skill(pool, display_title="Archive-Me", files=_FILES)
        await svc.archive_skill(pool, skill.id)

        skills = await svc.list_skills(pool, limit=100)
        assert skill.id not in [s.id for s in skills]

    async def test_create_requires_skill_md(self, pool: Any) -> None:
        from aios.errors import ValidationError
        from aios.services import skills as svc

        with pytest.raises(ValidationError, match=r"SKILL\.md"):
            await svc.create_skill(
                pool,
                display_title="Bad",
                files={"readme/README.md": "no skill file here"},
            )


class TestSkillVersions:
    async def test_create_new_version(self, pool: Any) -> None:
        from aios.services import skills as svc

        skill, v1 = await svc.create_skill(pool, display_title="Versioned", files=_FILES)
        assert v1.version == 1

        v2_files = {
            "test-skill/SKILL.md": "---\nname: test-skill\ndescription: Updated\n---\n# V2",
        }
        v2 = await svc.create_skill_version(pool, skill.id, files=v2_files)
        assert v2.version == 2
        assert v2.description == "Updated"

        updated = await svc.get_skill(pool, skill.id)
        assert updated.latest_version == 2

    async def test_list_versions(self, pool: Any) -> None:
        from aios.services import skills as svc

        skill, _ = await svc.create_skill(pool, display_title="Multi-V", files=_FILES)
        await svc.create_skill_version(
            pool,
            skill.id,
            files={"test-skill/SKILL.md": "---\nname: test-skill\n---\n# V2"},
        )
        versions = await svc.list_skill_versions(pool, skill.id)
        assert len(versions) == 2
        # Newest first
        assert versions[0].version == 2
        assert versions[1].version == 1

    async def test_get_specific_version(self, pool: Any) -> None:
        from aios.services import skills as svc

        skill, _ = await svc.create_skill(pool, display_title="Pin", files=_FILES)
        v1 = await svc.get_skill_version(pool, skill.id, 1)
        assert v1.version == 1
        assert v1.name == "test-skill"


class TestAgentSkills:
    async def test_agent_with_skills(self, pool: Any) -> None:
        from aios.services import agents as agents_svc
        from aios.services import skills as skills_svc

        skill, _ = await skills_svc.create_skill(pool, display_title="Agent-Skill", files=_FILES)

        agent = await agents_svc.create_agent(
            pool,
            name="skilled-agent",
            model="fake/test",
            system="test",
            tools=[],
            skills=[AgentSkillRef(skill_id=skill.id)],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        # Skills refs appear on agent with resolved version
        assert len(agent.skills) == 1
        assert agent.skills[0].skill_id == skill.id
        assert agent.skills[0].version == 1  # resolved from null → latest

    async def test_agent_without_skills(self, pool: Any) -> None:
        from aios.services import agents as agents_svc

        agent = await agents_svc.create_agent(
            pool,
            name="no-skills-agent",
            model="fake/test",
            system="test",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        assert agent.skills == []

    async def test_invalid_skill_ref_rejected(self, pool: Any) -> None:
        from aios.errors import NotFoundError
        from aios.services import agents as agents_svc

        with pytest.raises(NotFoundError):
            await agents_svc.create_agent(
                pool,
                name="bad-ref-agent",
                model="fake/test",
                system="test",
                tools=[],
                skills=[AgentSkillRef(skill_id="skl_NONEXISTENT")],
                description=None,
                metadata={},
                window_min=50_000,
                window_max=150_000,
            )

    async def test_update_agent_skills(self, pool: Any) -> None:
        from aios.services import agents as agents_svc
        from aios.services import skills as skills_svc

        skill, _ = await skills_svc.create_skill(
            pool,
            display_title="Upd-Skill",
            files={"upd-skill/SKILL.md": "---\nname: upd-skill\n---\n# Upd"},
        )
        agent = await agents_svc.create_agent(
            pool,
            name="update-skills-agent",
            model="fake/test",
            system="test",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        assert agent.skills == []

        updated = await agents_svc.update_agent(
            pool,
            agent.id,
            expected_version=agent.version,
            skills=[AgentSkillRef(skill_id=skill.id)],
        )
        assert len(updated.skills) == 1
        assert updated.skills[0].skill_id == skill.id

    async def test_skill_resolve(self, pool: Any) -> None:
        """resolve_skill_refs returns full SkillVersion objects."""
        from aios.services import skills as skills_svc

        skill, _ = await skills_svc.create_skill(
            pool,
            display_title="Resolve-Test",
            files={"resolve-test/SKILL.md": "---\nname: resolve-test\n---\n# Resolve"},
        )
        refs = [AgentSkillRef(skill_id=skill.id)]
        versions = await skills_svc.resolve_skill_refs(pool, refs)
        assert len(versions) == 1
        assert versions[0].name == "resolve-test"
        assert versions[0].directory == "resolve-test"
        assert "SKILL.md" in versions[0].files

    async def test_max_skills_limit(self, pool: Any) -> None:
        from aios.errors import ValidationError
        from aios.services import skills as skills_svc

        # Create 21 skills
        refs = []
        for i in range(21):
            s, _ = await skills_svc.create_skill(
                pool,
                display_title=f"Limit-{i}",
                files={f"limit-{i}/SKILL.md": f"---\nname: limit-{i}\n---\n# L{i}"},
            )
            refs.append(AgentSkillRef(skill_id=s.id))

        with pytest.raises(ValidationError, match="20"):
            await skills_svc.resolve_skill_refs(pool, refs)
