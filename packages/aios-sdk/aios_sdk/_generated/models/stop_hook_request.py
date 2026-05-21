from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.always_continue_stop_hook import AlwaysContinueStopHook
    from ..models.self_check_stop_hook import SelfCheckStopHook
    from ..models.task_call_stop_hook import TaskCallStopHook


T = TypeVar("T", bound="StopHookRequest")


@_attrs_define
class StopHookRequest:
    """Request body for ``POST /v1/sessions/{id}/stop-hook``.

    ``hook=None`` clears the hook (returns the session to conversational
    default).

        Attributes:
            hook (AlwaysContinueStopHook | None | SelfCheckStopHook | TaskCallStopHook | Unset):
    """

    hook: (
        AlwaysContinueStopHook | None | SelfCheckStopHook | TaskCallStopHook | Unset
    ) = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.always_continue_stop_hook import AlwaysContinueStopHook
        from ..models.self_check_stop_hook import SelfCheckStopHook
        from ..models.task_call_stop_hook import TaskCallStopHook

        hook: dict[str, Any] | None | Unset
        if isinstance(self.hook, Unset):
            hook = UNSET
        elif isinstance(self.hook, SelfCheckStopHook):
            hook = self.hook.to_dict()
        elif isinstance(self.hook, TaskCallStopHook):
            hook = self.hook.to_dict()
        elif isinstance(self.hook, AlwaysContinueStopHook):
            hook = self.hook.to_dict()
        else:
            hook = self.hook

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if hook is not UNSET:
            field_dict["hook"] = hook

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.always_continue_stop_hook import AlwaysContinueStopHook
        from ..models.self_check_stop_hook import SelfCheckStopHook
        from ..models.task_call_stop_hook import TaskCallStopHook

        d = dict(src_dict)

        def _parse_hook(
            data: object,
        ) -> (
            AlwaysContinueStopHook | None | SelfCheckStopHook | TaskCallStopHook | Unset
        ):
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                hook_type_0_type_0 = SelfCheckStopHook.from_dict(data)

                return hook_type_0_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                hook_type_0_type_1 = TaskCallStopHook.from_dict(data)

                return hook_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                hook_type_0_type_2 = AlwaysContinueStopHook.from_dict(data)

                return hook_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                AlwaysContinueStopHook
                | None
                | SelfCheckStopHook
                | TaskCallStopHook
                | Unset,
                data,
            )

        hook = _parse_hook(d.pop("hook", UNSET))

        stop_hook_request = cls(
            hook=hook,
        )

        return stop_hook_request
