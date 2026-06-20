from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.draft_streaming import DraftStreaming
    from ..models.native_buttons import NativeButtons


T = TypeVar("T", bound="ConnectorCapabilities")


@_attrs_define
class ConnectorCapabilities:
    """Typed richness descriptor — a ``tools_schema`` sibling on the catalog
    row.  Each field is a present/absent typed sub-object (a declared KIND),
    never a bool flag.  An absent field == capability not declared == the
    conservative rendering floor.

        Attributes:
            draft_streaming (DraftStreaming | None | Unset):
            native_buttons (NativeButtons | None | Unset):
    """

    draft_streaming: DraftStreaming | None | Unset = UNSET
    native_buttons: NativeButtons | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.draft_streaming import DraftStreaming
        from ..models.native_buttons import NativeButtons

        draft_streaming: dict[str, Any] | None | Unset
        if isinstance(self.draft_streaming, Unset):
            draft_streaming = UNSET
        elif isinstance(self.draft_streaming, DraftStreaming):
            draft_streaming = self.draft_streaming.to_dict()
        else:
            draft_streaming = self.draft_streaming

        native_buttons: dict[str, Any] | None | Unset
        if isinstance(self.native_buttons, Unset):
            native_buttons = UNSET
        elif isinstance(self.native_buttons, NativeButtons):
            native_buttons = self.native_buttons.to_dict()
        else:
            native_buttons = self.native_buttons

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if draft_streaming is not UNSET:
            field_dict["draft_streaming"] = draft_streaming
        if native_buttons is not UNSET:
            field_dict["native_buttons"] = native_buttons

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.draft_streaming import DraftStreaming
        from ..models.native_buttons import NativeButtons

        d = dict(src_dict)

        def _parse_draft_streaming(data: object) -> DraftStreaming | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                draft_streaming_type_0 = DraftStreaming.from_dict(data)

                return draft_streaming_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(DraftStreaming | None | Unset, data)

        draft_streaming = _parse_draft_streaming(d.pop("draft_streaming", UNSET))

        def _parse_native_buttons(data: object) -> NativeButtons | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                native_buttons_type_0 = NativeButtons.from_dict(data)

                return native_buttons_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(NativeButtons | None | Unset, data)

        native_buttons = _parse_native_buttons(d.pop("native_buttons", UNSET))

        connector_capabilities = cls(
            draft_streaming=draft_streaming,
            native_buttons=native_buttons,
        )

        return connector_capabilities
