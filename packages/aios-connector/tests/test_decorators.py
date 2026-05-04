"""@tool / @focal_required decorator tests.

These exercise the static surface — schema generation, descriptor
attachment, focal flag, signature-inspection-driven focal kwargs —
without spinning up an MCP server.  End-to-end dispatch is covered
by the e2e suite via the echo connector.
"""

from __future__ import annotations

import pytest
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
    async def single_account_focal(self, *, chat_id: str) -> dict[str, str]:
        """Single-account focal tool — chat_id only, no account kwarg."""
        return {"chat_id": chat_id}

    @tool()
    @focal_required
    async def multi_account_focal(self, *, account: str, chat_id: str) -> dict[str, str]:
        """Multi-account focal tool — both account and chat_id injected."""
        return {"account": account, "chat_id": chat_id}

    @tool()
    @focal_required
    async def account_only_focal(self, *, account: str) -> dict[str, str]:
        """Focal tool taking only account (uncommon but supported)."""
        return {"account": account}

    @tool()
    async def explicit_account_arg(self, account: str) -> dict[str, str]:
        """Non-focal tool with model-visible account arg (per design §3.4)."""
        return {"account": account}

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


def test_single_account_focal_kwargs() -> None:
    descriptor = _descriptor(_FixtureConnector.single_account_focal)
    assert descriptor.focal_required is True
    assert descriptor.focal_kwargs == frozenset({"chat_id"})
    # Both injected kwargs are excluded from the schema since focal_required.
    assert "chat_id" not in descriptor.input_schema["properties"]
    assert "account" not in descriptor.input_schema["properties"]


def test_multi_account_focal_kwargs() -> None:
    descriptor = _descriptor(_FixtureConnector.multi_account_focal)
    assert descriptor.focal_required is True
    assert descriptor.focal_kwargs == frozenset({"account", "chat_id"})
    assert descriptor.input_schema["properties"] == {}


def test_account_only_focal_kwargs() -> None:
    descriptor = _descriptor(_FixtureConnector.account_only_focal)
    assert descriptor.focal_required is True
    assert descriptor.focal_kwargs == frozenset({"account"})


def test_non_focal_account_arg_stays_in_schema() -> None:
    """Per design §3.4: non-focal tools like list_chats(account: str) keep
    account as a model-visible argument."""
    descriptor = _descriptor(_FixtureConnector.explicit_account_arg)
    assert descriptor.focal_required is False
    assert descriptor.focal_kwargs == frozenset()
    properties = descriptor.input_schema["properties"]
    assert properties["account"] == {"type": "string"}
    assert descriptor.input_schema["required"] == ["account"]


def test_non_focal_tool_has_focal_required_false() -> None:
    descriptor = _descriptor(_FixtureConnector.no_args)
    assert descriptor.focal_required is False
    assert descriptor.focal_kwargs == frozenset()


def test_explicit_name_overrides_method_name() -> None:
    descriptor = _descriptor(_FixtureConnector.actually_called_renamed)
    assert descriptor.name == "renamed"


def test_focal_required_without_injectable_kwarg_raises() -> None:
    """A @focal_required tool with neither account nor chat_id is a bug.

    The SDK injects only what's in the signature; a tool that wants
    neither shouldn't be focal-required (use @tool() alone instead).
    """
    with pytest.raises(ValueError, match="must declare at least one of"):

        class _BadConnector(Connector):  # pyright: ignore[reportUnusedClass]
            name = "_bad"

            @tool()
            @focal_required
            async def empty_focal(self, text: str) -> dict[str, str]:
                """Mistakenly focal-required without injectable kwarg."""
                return {"text": text}


def test_collect_tools_picks_up_all_decorated(tmp_path: object) -> None:
    from pathlib import Path

    connector = _FixtureConnector(spool_dir=Path(str(tmp_path)))
    names = sorted(t.name for t in connector._tools)
    assert names == sorted(
        [
            "no_args",
            "with_args",
            "single_account_focal",
            "multi_account_focal",
            "account_only_focal",
            "explicit_account_arg",
            "renamed",
        ]
    )


def test_make_account_minimal() -> None:
    assert make_account("a-1", "Account One") == {"id": "a-1", "display_name": "Account One"}


def test_make_account_with_metadata() -> None:
    payload = make_account("a-1", "Account One", {"phone": "+1555"})
    assert payload == {
        "id": "a-1",
        "display_name": "Account One",
        "metadata": {"phone": "+1555"},
    }
