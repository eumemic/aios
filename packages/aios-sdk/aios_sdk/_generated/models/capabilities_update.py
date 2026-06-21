from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.connector_capabilities import ConnectorCapabilities


T = TypeVar("T", bound="CapabilitiesUpdate")


@_attrs_define
class CapabilitiesUpdate:
    """Body for ``PUT /v1/connectors/{connector}/capabilities``.

    A sibling to :class:`ToolsSchemaUpdate` — kept separate so capability churn
    is decoupled from a full ``tools_schema`` republish and the shipped
    ``tools_schema`` body contract stays untouched.

        Attributes:
            capabilities (ConnectorCapabilities): Typed richness descriptor — a ``tools_schema`` sibling on the catalog
                row.  Each field is a present/absent typed sub-object (a declared KIND),
                never a bool flag.  An absent field == capability not declared == the
                conservative rendering floor.
    """

    capabilities: ConnectorCapabilities

    def to_dict(self) -> dict[str, Any]:
        capabilities = self.capabilities.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "capabilities": capabilities,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connector_capabilities import ConnectorCapabilities

        d = dict(src_dict)
        capabilities = ConnectorCapabilities.from_dict(d.pop("capabilities"))

        capabilities_update = cls(
            capabilities=capabilities,
        )

        return capabilities_update
