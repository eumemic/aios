from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="OAuthCompleteRequest")


@_attrs_define
class OAuthCompleteRequest:
    """Finish an interactive OAuth flow: exchange the returned code for tokens.

    The ``state`` correlates back to the in-progress flow (and guards CSRF);
    ``code`` is the authorization code the provider returned to the callback.

        Attributes:
            state (str):
            code (str):
    """

    state: str
    code: str

    def to_dict(self) -> dict[str, Any]:
        state = self.state

        code = self.code

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "state": state,
                "code": code,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        state = d.pop("state")

        code = d.pop("code")

        o_auth_complete_request = cls(
            state=state,
            code=code,
        )

        return o_auth_complete_request
