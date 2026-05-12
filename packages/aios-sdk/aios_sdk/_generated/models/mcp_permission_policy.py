from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.mcp_permission_policy_type import McpPermissionPolicyType

T = TypeVar("T", bound="McpPermissionPolicy")


@_attrs_define
class McpPermissionPolicy:
    """Wrapper matching Anthropic's ``{type: "always_allow"}`` shape.

    Attributes:
        type_ (McpPermissionPolicyType):
    """

    type_: McpPermissionPolicyType

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = McpPermissionPolicyType(d.pop("type"))

        mcp_permission_policy = cls(
            type_=type_,
        )

        return mcp_permission_policy
