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

T = TypeVar("T", bound="SandboxCommandAction")


@_attrs_define
class SandboxCommandAction:
    """Run a bash command in the session's sandbox WITHOUT waking the model.

    Defaults for ``timeout_seconds`` / ``max_output_bytes`` are materialized
    at write time so the stored row is self-describing (the runner carries no
    defaults knowledge).

        Attributes:
            command (str):
            kind (Literal['sandbox_command'] | Unset):  Default: 'sandbox_command'.
            timeout_seconds (int | Unset):  Default: 300.
            max_output_bytes (int | Unset):  Default: 65536.
    """

    command: str
    kind: Literal["sandbox_command"] | Unset = "sandbox_command"
    timeout_seconds: int | Unset = 300
    max_output_bytes: int | Unset = 65536

    def to_dict(self) -> dict[str, Any]:
        command = self.command

        kind = self.kind

        timeout_seconds = self.timeout_seconds

        max_output_bytes = self.max_output_bytes

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "command": command,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind
        if timeout_seconds is not UNSET:
            field_dict["timeout_seconds"] = timeout_seconds
        if max_output_bytes is not UNSET:
            field_dict["max_output_bytes"] = max_output_bytes

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        command = d.pop("command")

        kind = cast(Literal["sandbox_command"] | Unset, d.pop("kind", UNSET))
        if kind != "sandbox_command" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'sandbox_command', got '{kind}'")

        timeout_seconds = d.pop("timeout_seconds", UNSET)

        max_output_bytes = d.pop("max_output_bytes", UNSET)

        sandbox_command_action = cls(
            command=command,
            kind=kind,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

        return sandbox_command_action
