from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.ssh_permission_policy_type import SshPermissionPolicyType

T = TypeVar("T", bound="SshPermissionPolicy")


@_attrs_define
class SshPermissionPolicy:
    """Wrapper matching the ``{type: "always_allow"}`` shape used for MCP/HTTP.

    Attributes:
        type_ (SshPermissionPolicyType):
    """

    type_: SshPermissionPolicyType

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
        type_ = SshPermissionPolicyType(d.pop("type"))

        ssh_permission_policy = cls(
            type_=type_,
        )

        return ssh_permission_policy
