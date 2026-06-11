from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SandboxCommandActionReplace")


@_attrs_define
class SandboxCommandActionReplace:
    """Update-side variant: optional-at-create fields are REQUIRED, so a
    partial action 422s instead of silently resetting stored values to
    defaults. (Create keeps the defaults for tool ergonomics.)

        Attributes:
            command (str):
            timeout_seconds (int):
            max_output_bytes (int):
            kind (Literal['sandbox_command'] | Unset):  Default: 'sandbox_command'.
    """

    command: str
    timeout_seconds: int
    max_output_bytes: int
    kind: Literal["sandbox_command"] | Unset = "sandbox_command"

    def to_dict(self) -> dict[str, Any]:
        command = self.command

        timeout_seconds = self.timeout_seconds

        max_output_bytes = self.max_output_bytes

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "command": command,
                "timeout_seconds": timeout_seconds,
                "max_output_bytes": max_output_bytes,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        command = d.pop("command")

        timeout_seconds = d.pop("timeout_seconds")

        max_output_bytes = d.pop("max_output_bytes")

        kind = cast(Literal["sandbox_command"] | Unset, d.pop("kind", UNSET))
        if kind != "sandbox_command" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'sandbox_command', got '{kind}'")

        sandbox_command_action_replace = cls(
            command=command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            kind=kind,
        )

        return sandbox_command_action_replace
