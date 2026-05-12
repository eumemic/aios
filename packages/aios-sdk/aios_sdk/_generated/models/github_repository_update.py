from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="GithubRepositoryUpdate")


@_attrs_define
class GithubRepositoryUpdate:
    """Request body for ``POST /v1/sessions/{sid}/resources/{rid}``.

    Token rotation is the primary action; ``git_user_name`` /
    ``git_user_email`` may also be supplied alongside, in which case the
    stored identity is replaced.  ``url`` and ``mount_path`` are
    immutable after creation — to change them, detach the resource and
    attach a new one.

        Attributes:
            authorization_token (str):
            git_user_name (None | str | Unset):
            git_user_email (None | str | Unset):
    """

    authorization_token: str
    git_user_name: None | str | Unset = UNSET
    git_user_email: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
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

        github_repository_update = cls(
            authorization_token=authorization_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )

        return github_repository_update
