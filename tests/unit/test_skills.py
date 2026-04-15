"""Unit tests for the skills system.

Tests the pure functions: system prompt augmentation, frontmatter
parsing, model validation, and file provisioning logic.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pydantic
import pytest

from aios.errors import ValidationError
from aios.harness.skills import (
    augment_system_prompt,
    build_skills_system_block,
    provision_skill_files,
)
from aios.models.skills import AgentSkillRef, SkillCreate, SkillVersion, SkillVersionCreate
from aios.services.skills import (
    _extract_skill_metadata,
    _validate_skill_name,
    parse_skill_md,
    serialize_skills_for_snapshot,
)

# ── helpers ────────────────────────────────────────────────────────────────

_NOW = datetime.now(tz=UTC)


def _sv(
    name: str = "code-review",
    directory: str = "code-review",
    description: str = "Expert code review skill",
    version: int = 1,
    skill_id: str = "skl_01TEST",
) -> SkillVersion:
    return SkillVersion(
        skill_id=skill_id,
        version=version,
        directory=directory,
        name=name,
        description=description,
        files={"SKILL.md": f"---\nname: {name}\ndescription: {description}\n---\n# Instructions"},
        created_at=_NOW,
    )


# ── TestBuildSkillsSystemBlock ─────────────────────────────────────────────


class TestBuildSkillsSystemBlock:
    def test_empty_skills_returns_empty(self) -> None:
        assert build_skills_system_block([]) == ""

    def test_single_skill_contains_xml(self) -> None:
        result = build_skills_system_block([_sv()])
        assert "<available_skills>" in result
        assert "<name>code-review</name>" in result
        assert "<description>Expert code review skill</description>" in result
        assert "<location>/workspace/skills/code-review/SKILL.md</location>" in result
        assert "</available_skills>" in result

    def test_contains_skills_policy(self) -> None:
        result = build_skills_system_block([_sv()])
        assert "# Skills usage policy" in result
        assert "read tool" in result

    def test_multiple_skills_ordering(self) -> None:
        sv1 = _sv(name="alpha", directory="alpha")
        sv2 = _sv(name="beta", directory="beta")
        result = build_skills_system_block([sv1, sv2])
        assert "<name>alpha</name>" in result
        assert "<name>beta</name>" in result
        assert result.index("alpha") < result.index("beta")

    def test_location_uses_directory(self) -> None:
        result = build_skills_system_block([_sv(directory="my-custom-skill")])
        assert "/workspace/skills/my-custom-skill/SKILL.md" in result


# ── TestAugmentSystemPrompt ────────────────────────────────────────────────


class TestAugmentSystemPrompt:
    def test_no_skills_returns_base(self) -> None:
        assert augment_system_prompt("You are helpful.", []) == "You are helpful."

    def test_appends_to_base(self) -> None:
        result = augment_system_prompt("You are helpful.", [_sv()])
        assert result.startswith("You are helpful.\n\n")
        assert "<available_skills>" in result

    def test_empty_base_with_skills(self) -> None:
        result = augment_system_prompt("", [_sv()])
        assert result.startswith("# Skills usage policy")

    def test_no_skills_empty_base(self) -> None:
        assert augment_system_prompt("", []) == ""


# ── TestParseSkillMd ───────────────────────────────────────────────────────


class TestParseSkillMd:
    def test_basic_frontmatter(self) -> None:
        content = "---\nname: code-review\ndescription: Review code\n---\n# Body"
        name, desc = parse_skill_md(content)
        assert name == "code-review"
        assert desc == "Review code"

    def test_name_only(self) -> None:
        content = "---\nname: my-skill\n---\n# First heading"
        result_name, desc = parse_skill_md(content)
        assert result_name == "my-skill"
        assert desc == "First heading"

    def test_description_fallback_to_body(self) -> None:
        content = "---\nname: test\n---\n\nSome description line"
        _name, desc = parse_skill_md(content)
        assert desc == "Some description line"

    def test_missing_frontmatter_raises(self) -> None:
        with pytest.raises(ValidationError, match="frontmatter"):
            parse_skill_md("# No frontmatter here")

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            parse_skill_md("---\ndescription: no name\n---\n# Body")

    def test_unclosed_frontmatter_raises(self) -> None:
        with pytest.raises(ValidationError, match="closing"):
            parse_skill_md("---\nname: test\n# Body")

    def test_quoted_values_stripped(self) -> None:
        content = "---\nname: 'my-skill'\ndescription: \"A skill\"\n---\n"
        name, desc = parse_skill_md(content)
        assert name == "my-skill"
        assert desc == "A skill"

    def test_leading_newlines_ok(self) -> None:
        content = "\n\n---\nname: test\n---\n# Body"
        result_name, _desc = parse_skill_md(content)
        assert result_name == "test"


# ── TestValidateSkillName ──────────────────────────────────────────────────


class TestValidateSkillName:
    def test_valid_names(self) -> None:
        for name in ("code-review", "my-skill-123", "a", "test"):
            _validate_skill_name(name)  # should not raise

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate_skill_name("Code-Review")

    def test_underscores_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate_skill_name("code_review")

    def test_too_long(self) -> None:
        with pytest.raises(ValidationError, match="64"):
            _validate_skill_name("a" * 65)

    def test_starts_with_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate_skill_name("-starts-bad")


# ── TestExtractSkillMetadata ───────────────────────────────────────────────


class TestExtractSkillMetadata:
    def test_basic_extraction(self) -> None:
        files = {
            "code-review/SKILL.md": "---\nname: code-review\ndescription: Review\n---\n# Go",
            "code-review/scripts/lint.py": "print('lint')",
        }
        directory, name, desc, normalized = _extract_skill_metadata(files)
        assert directory == "code-review"
        assert name == "code-review"
        assert desc == "Review"
        assert "SKILL.md" in normalized
        assert "scripts/lint.py" in normalized
        assert "code-review/SKILL.md" not in normalized

    def test_no_skill_md_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"SKILL\.md"):
            _extract_skill_metadata({"readme.md": "hello"})

    def test_bare_skill_md_raises(self) -> None:
        with pytest.raises(ValidationError, match="directory"):
            _extract_skill_metadata({"SKILL.md": "---\nname: test\n---\n"})

    def test_multiple_skill_md_raises(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            _extract_skill_metadata(
                {
                    "a/SKILL.md": "---\nname: a\n---\n",
                    "b/SKILL.md": "---\nname: b\n---\n",
                }
            )


# ── TestSkillModels ────────────────────────────────────────────────────────


class TestSkillModels:
    def test_agent_skill_ref_defaults(self) -> None:
        ref = AgentSkillRef(skill_id="skl_01TEST")
        assert ref.version is None

    def test_agent_skill_ref_pinned(self) -> None:
        ref = AgentSkillRef(skill_id="skl_01TEST", version=3)
        assert ref.version == 3

    def test_skill_create_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            SkillCreate(display_title="Test", files={}, bogus="extra")  # type: ignore[call-arg]

    def test_skill_version_create_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            SkillVersionCreate(files={}, bogus="extra")  # type: ignore[call-arg]


# ── TestSerializeSkillsForSnapshot ─────────────────────────────────────────


class TestSerializeSkillsForSnapshot:
    def test_resolves_null_versions(self) -> None:
        refs = [AgentSkillRef(skill_id="skl_01A"), AgentSkillRef(skill_id="skl_01B", version=2)]
        versions = [_sv(skill_id="skl_01A", version=5), _sv(skill_id="skl_01B", version=2)]
        result = serialize_skills_for_snapshot(refs, versions)
        parsed = json.loads(result)
        assert parsed == [
            {"skill_id": "skl_01A", "version": 5},
            {"skill_id": "skl_01B", "version": 2},
        ]


# ── TestProvisionSkillFiles ────────────────────────────────────────────────


class TestProvisionSkillFiles:
    @pytest.mark.asyncio
    async def test_writes_files_to_workspace(self) -> None:
        sv = SkillVersion(
            skill_id="skl_01TEST",
            version=1,
            directory="my-skill",
            name="my-skill",
            description="Test skill",
            files={
                "SKILL.md": "# Instructions",
                "scripts/run.py": "print('hello')",
            },
            created_at=_NOW,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "sess_01TEST"
            with (
                patch(
                    "aios.harness.skills._load_workspace_path",
                    AsyncMock(return_value=str(workspace)),
                ),
                patch("aios.harness.skills.ensure_workspace_path", return_value=workspace),
            ):
                await provision_skill_files("sess_01TEST", [sv])

            skill_md = workspace / "skills" / "my-skill" / "SKILL.md"
            assert skill_md.exists()
            assert skill_md.read_text() == "# Instructions"

            script = workspace / "skills" / "my-skill" / "scripts" / "run.py"
            assert script.exists()
            assert script.read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_idempotent_skips_if_exists(self) -> None:
        sv = _sv()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "sess_01TEST"
            skills_dir = workspace / "skills"
            skills_dir.mkdir(parents=True)
            with (
                patch(
                    "aios.harness.skills._load_workspace_path",
                    AsyncMock(return_value=str(workspace)),
                ),
                patch("aios.harness.skills.ensure_workspace_path", return_value=workspace),
            ):
                await provision_skill_files("sess_01TEST", [sv])

            assert not (skills_dir / "code-review").exists()

    @pytest.mark.asyncio
    async def test_empty_list_noop(self) -> None:
        await provision_skill_files("sess_01TEST", [])
