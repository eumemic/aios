from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.wf_run_wait_response_run_status import WfRunWaitResponseRunStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.wf_run_wait_response_error_type_0 import WfRunWaitResponseErrorType0


T = TypeVar("T", bound="WfRunWaitResponse")


@_attrs_define
class WfRunWaitResponse:
    """Response for ``GET /v1/runs/{run_id}/wait`` — the run's completion record, or its
    current (non-terminal) state if the wait timed out.

    Deliberately mirrors the ``{result, is_error, error}`` shape of a request response
    (``derive_response``) so the ``await`` primitive's two backings (run-terminal and, later,
    session-request) share one envelope. Poll until ``done``: a still-running run returns
    ``done=False`` with its live ``run_status`` (``running``/``suspended``/…); call again to
    keep blocking.

        Attributes:
            run_status (WfRunWaitResponseRunStatus):
            done (bool):
            output (Any | Unset):
            is_error (bool | Unset):  Default: False.
            error (None | Unset | WfRunWaitResponseErrorType0):
    """

    run_status: WfRunWaitResponseRunStatus
    done: bool
    output: Any | Unset = UNSET
    is_error: bool | Unset = False
    error: None | Unset | WfRunWaitResponseErrorType0 = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.wf_run_wait_response_error_type_0 import (
            WfRunWaitResponseErrorType0,
        )

        run_status = self.run_status.value

        done = self.done

        output = self.output

        is_error = self.is_error

        error: dict[str, Any] | None | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        elif isinstance(self.error, WfRunWaitResponseErrorType0):
            error = self.error.to_dict()
        else:
            error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "run_status": run_status,
                "done": done,
            }
        )
        if output is not UNSET:
            field_dict["output"] = output
        if is_error is not UNSET:
            field_dict["is_error"] = is_error
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.wf_run_wait_response_error_type_0 import (
            WfRunWaitResponseErrorType0,
        )

        d = dict(src_dict)
        run_status = WfRunWaitResponseRunStatus(d.pop("run_status"))

        done = d.pop("done")

        output = d.pop("output", UNSET)

        is_error = d.pop("is_error", UNSET)

        def _parse_error(data: object) -> None | Unset | WfRunWaitResponseErrorType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                error_type_0 = WfRunWaitResponseErrorType0.from_dict(data)

                return error_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WfRunWaitResponseErrorType0, data)

        error = _parse_error(d.pop("error", UNSET))

        wf_run_wait_response = cls(
            run_status=run_status,
            done=done,
            output=output,
            is_error=is_error,
            error=error,
        )

        wf_run_wait_response.additional_properties = d
        return wf_run_wait_response

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
