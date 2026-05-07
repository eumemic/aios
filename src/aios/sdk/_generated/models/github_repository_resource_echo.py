from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="GithubRepositoryResourceEcho")


@_attrs_define
class GithubRepositoryResourceEcho:
    """Read view of an attached github repository as echoed on
    ``Session.resources`` and on the per-resource sub-collection endpoints.

    Token is intentionally absent; ``id`` is the per-attachment ULID
    (prefix ``ghrepo``) used by the rotation endpoint.  ``git_user_name``
    / ``git_user_email`` echo back as plaintext (they aren't secrets).

        Attributes:
            id (str):
            url (str):
            mount_path (str):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            type_ (Literal['github_repository'] | Unset):  Default: 'github_repository'.
            git_user_name (None | str | Unset):
            git_user_email (None | str | Unset):
    """

    id: str
    url: str
    mount_path: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    type_: Literal["github_repository"] | Unset = "github_repository"
    git_user_name: None | str | Unset = UNSET
    git_user_email: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        url = self.url

        mount_path = self.mount_path

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        type_ = self.type_

        git_user_name: None | str | Unset
        if isinstance(self.git_user_name, Unset):
            git_user_name = UNSET
        else:
            git_user_name = self.git_user_name

        git_user_email: None | str | Unset
        if isinstance(self.git_user_email, Unset):
            git_user_email = UNSET
        else:
            git_user_email = self.git_user_email

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "url": url,
                "mount_path": mount_path,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if git_user_name is not UNSET:
            field_dict["git_user_name"] = git_user_name
        if git_user_email is not UNSET:
            field_dict["git_user_email"] = git_user_email

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        url = d.pop("url")

        mount_path = d.pop("mount_path")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        type_ = cast(Literal["github_repository"] | Unset, d.pop("type", UNSET))
        if type_ != "github_repository" and not isinstance(type_, Unset):
            raise ValueError(
                f"type must match const 'github_repository', got '{type_}'"
            )

        def _parse_git_user_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        git_user_name = _parse_git_user_name(d.pop("git_user_name", UNSET))

        def _parse_git_user_email(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        git_user_email = _parse_git_user_email(d.pop("git_user_email", UNSET))

        github_repository_resource_echo = cls(
            id=id,
            url=url,
            mount_path=mount_path,
            created_at=created_at,
            updated_at=updated_at,
            type_=type_,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )

        github_repository_resource_echo.additional_properties = d
        return github_repository_resource_echo

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
