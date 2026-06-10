from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="GateResume")


@_attrs_define
class GateResume:
    """Request body for ``POST /v1/runs/{run_id}/resume`` — deliver a gate's value.

    Keyed by ``gate_nonce`` (the unguessable capability token minted into the gate's
    ``call_started`` event), not the internal ``call_key``. ``result`` is the
    externally-delivered resume value (arbitrary JSON).

        Attributes:
            gate_nonce (str):
            result (Any | Unset):
    """

    gate_nonce: str
    result: Any | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        gate_nonce = self.gate_nonce

        result = self.result

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "gate_nonce": gate_nonce,
            }
        )
        if result is not UNSET:
            field_dict["result"] = result

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        gate_nonce = d.pop("gate_nonce")

        result = d.pop("result", UNSET)

        gate_resume = cls(
            gate_nonce=gate_nonce,
            result=result,
        )

        return gate_resume
