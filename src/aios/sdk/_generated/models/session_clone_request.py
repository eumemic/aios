from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionCloneRequest")


@_attrs_define
class SessionCloneRequest:
    """Request body for ``POST /v1/sessions/{id}/clone``.

    All fields optional; the clone inherits everything not overridden from
    the parent at clone time.

        Attributes:
            workspace_path (None | str | Unset): Override the clone's workspace volume path. Defaults to a fresh
                ``workspace_root/<new_session_id>`` so clones don't fight over files. The directory must exist; aios will not
                create it.
    """

    workspace_path: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        workspace_path: None | str | Unset
        if isinstance(self.workspace_path, Unset):
            workspace_path = UNSET
        else:
            workspace_path = self.workspace_path

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if workspace_path is not UNSET:
            field_dict["workspace_path"] = workspace_path

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_workspace_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        workspace_path = _parse_workspace_path(d.pop("workspace_path", UNSET))

        session_clone_request = cls(
            workspace_path=workspace_path,
        )

        return session_clone_request
