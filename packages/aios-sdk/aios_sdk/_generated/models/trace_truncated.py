from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TraceTruncated")


@_attrs_define
class TraceTruncated:
    """The typed marker that the walk hit its node-count ceiling.

    ``#1124``'s depth counter bounds path length (≤10 by construction), but it
    does NOT bound the node count: ``workflow_max_agent_calls`` defaults to 1000
    lifetime ``agent()`` children per run. So the walk carries an explicit
    config-tunable node-count ceiling; when it trips, ``at_nodes`` records how
    many nodes were emitted before the frontier was cut. The response stays
    well-formed (a partial-but-honest tree), never a silent truncation.

        Attributes:
            at_nodes (int): Number of nodes emitted before the ceiling cut the walk.
    """

    at_nodes: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at_nodes = self.at_nodes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "at_nodes": at_nodes,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        at_nodes = d.pop("at_nodes")

        trace_truncated = cls(
            at_nodes=at_nodes,
        )

        trace_truncated.additional_properties = d
        return trace_truncated

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
