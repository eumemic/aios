from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.account_config import AccountConfig


T = TypeVar("T", bound="UpdateAccountRequest")


@_attrs_define
class UpdateAccountRequest:
    """Body for ``PATCH /v1/accounts/{id}``.

    Partial update: omitted fields are preserved. All fields are optional;
    submitting none is a no-op that returns the account row unchanged.
    ``config`` is *merged* into the stored config (only the keys present in
    the submitted object are written), so setting one config item never
    disturbs the others.

        Attributes:
            display_name (None | str | Unset):
            can_mint_children (bool | None | Unset):
            config (AccountConfig | None | Unset):
    """

    display_name: None | str | Unset = UNSET
    can_mint_children: bool | None | Unset = UNSET
    config: AccountConfig | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.account_config import AccountConfig

        display_name: None | str | Unset
        if isinstance(self.display_name, Unset):
            display_name = UNSET
        else:
            display_name = self.display_name

        can_mint_children: bool | None | Unset
        if isinstance(self.can_mint_children, Unset):
            can_mint_children = UNSET
        else:
            can_mint_children = self.can_mint_children

        config: dict[str, Any] | None | Unset
        if isinstance(self.config, Unset):
            config = UNSET
        elif isinstance(self.config, AccountConfig):
            config = self.config.to_dict()
        else:
            config = self.config

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if can_mint_children is not UNSET:
            field_dict["can_mint_children"] = can_mint_children
        if config is not UNSET:
            field_dict["config"] = config

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.account_config import AccountConfig

        d = dict(src_dict)

        def _parse_display_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        display_name = _parse_display_name(d.pop("display_name", UNSET))

        def _parse_can_mint_children(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        can_mint_children = _parse_can_mint_children(d.pop("can_mint_children", UNSET))

        def _parse_config(data: object) -> AccountConfig | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                config_type_0 = AccountConfig.from_dict(data)

                return config_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AccountConfig | None | Unset, data)

        config = _parse_config(d.pop("config", UNSET))

        update_account_request = cls(
            display_name=display_name,
            can_mint_children=can_mint_children,
            config=config,
        )

        return update_account_request
