"""@tool / @focal_required decorator tests.

These exercise the static surface — schema generation, descriptor
attachment, focal flag — without spinning up an MCP server.
End-to-end dispatch is covered by the e2e suite via the echo
connector.
"""

from __future__ import annotations

from aios_connector import Connector, focal_required, make_account, tool
from aios_connector.base import _TOOL_ATTR, ToolDescriptor


class _FixtureConnector(Connector):
    name = "_fixture"

    @tool()
    async def no_args(self) -> dict[str, str]:
        """Tool with no arguments."""
        return {"ok": "yes"}

    @tool()
    async def with_args(self, text: str, count: int = 1) -> dict[str, object]:
        """Tool with a required and an optional argument."""
        return {"text": text, "count": count}

    @tool()
    @focal_required
    async def needs_focal(self, *, focal: str) -> dict[str, str]:
        """Tool requiring focal channel injection."""
        return {"focal": focal}

    @tool(name="renamed")
    async def actually_called_renamed(self) -> dict[str, str]:
        """Tool with an explicit name override."""
        return {"name": "renamed"}


def _descriptor(fn: object) -> ToolDescriptor:
    descriptor = getattr(fn, _TOOL_ATTR, None)
    assert isinstance(descriptor, ToolDescriptor)
    return descriptor


def test_no_args_schema_has_no_required() -> None:
    descriptor = _descriptor(_FixtureConnector.no_args)
    assert descriptor.input_schema["type"] == "object"
    assert descriptor.input_schema["properties"] == {}
    assert "required" not in descriptor.input_schema


def test_with_args_schema_required_and_optional() -> None:
    descriptor = _descriptor(_FixtureConnector.with_args)
    properties = descriptor.input_schema["properties"]
    assert properties["text"] == {"type": "string"}
    assert properties["count"] == {"type": "integer"}
    assert descriptor.input_schema["required"] == ["text"]


def test_focal_required_flag() -> None:
    descriptor = _descriptor(_FixtureConnector.needs_focal)
    assert descriptor.focal_required is True
    # The injected focal kwarg must NOT appear in the published schema —
    # only model-supplied args do.
    assert "focal" not in descriptor.input_schema["properties"]


def test_non_focal_tool_has_focal_required_false() -> None:
    descriptor = _descriptor(_FixtureConnector.no_args)
    assert descriptor.focal_required is False


def test_explicit_name_overrides_method_name() -> None:
    descriptor = _descriptor(_FixtureConnector.actually_called_renamed)
    assert descriptor.name == "renamed"


def test_collect_tools_picks_up_all_decorated(tmp_path: object) -> None:
    from pathlib import Path

    connector = _FixtureConnector(spool_dir=Path(str(tmp_path)))
    names = sorted(t.name for t in connector._tools)
    assert names == sorted(["no_args", "with_args", "needs_focal", "renamed"])


def test_make_account_minimal() -> None:
    assert make_account("a-1", "Account One") == {"id": "a-1", "display_name": "Account One"}


def test_make_account_with_metadata() -> None:
    payload = make_account("a-1", "Account One", {"phone": "+1555"})
    assert payload == {
        "id": "a-1",
        "display_name": "Account One",
        "metadata": {"phone": "+1555"},
    }
