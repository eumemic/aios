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

T = TypeVar("T", bound="GithubRepositoryResource")


@_attrs_define
class GithubRepositoryResource:
    """Item in ``Session.resources[]`` request shape (create + update).

    The ``authorization_token`` is a write-only ``SecretStr`` — present on
    request, encrypted on the way to the DB, and never returned in API
    responses (see :class:`GithubRepositoryResourceEcho`).

    ``git_user_name`` / ``git_user_email`` are optional and stamped via
    ``git config`` after clone so commits made inside the sandbox carry
    a deterministic identity per resource without the agent needing to
    self-correct from git's "Please tell me who you are" error.  Absent
    means "no identity configured" — the v1 default.

        Attributes:
            type_ (Literal['github_repository']):
            url (str):
            mount_path (str):
            authorization_token (str):
            git_user_name (None | str | Unset):
            git_user_email (None | str | Unset):
    """

    type_: Literal["github_repository"]
    url: str
    mount_path: str
    authorization_token: str
    git_user_name: None | str | Unset = UNSET
    git_user_email: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        url = self.url

        mount_path = self.mount_path

        authorization_token = self.authorization_token

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

        field_dict.update(
            {
                "type": type_,
                "url": url,
                "mount_path": mount_path,
                "authorization_token": authorization_token,
            }
        )
        if git_user_name is not UNSET:
            field_dict["git_user_name"] = git_user_name
        if git_user_email is not UNSET:
            field_dict["git_user_email"] = git_user_email

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["github_repository"], d.pop("type"))
        if type_ != "github_repository":
            raise ValueError(
                f"type must match const 'github_repository', got '{type_}'"
            )

        url = d.pop("url")

        mount_path = d.pop("mount_path")

        authorization_token = d.pop("authorization_token")

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

        github_repository_resource = cls(
            type_=type_,
            url=url,
            mount_path=mount_path,
            authorization_token=authorization_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )

        return github_repository_resource
