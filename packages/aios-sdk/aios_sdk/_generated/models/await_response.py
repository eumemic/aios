from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.await_response_outcome_type_0 import AwaitResponseOutcomeType0
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.await_response_error_type_0 import AwaitResponseErrorType0


T = TypeVar("T", bound="AwaitResponse")


@_attrs_define
class AwaitResponse:
    """The one completion envelope — ``GET /v1/tasks/{task_id}/await``.

    Unifies the session and run completion long-polls. ``outcome`` is the
    terminal state minus liveness (the trace's ``TerminalState`` with
    ``suspended``/``running`` folded into pending): ``None`` means **still
    pending** — the long-poll timed out before the task reached a terminal
    state, so re-poll. ``result`` carries the servicer's return value on ``ok``;
    ``error`` carries the ``{kind, message, …}`` detail on ``errored`` /
    ``cancelled``.

        Attributes:
            outcome (AwaitResponseOutcomeType0 | None | Unset): The task's terminal outcome, or null while it is still
                pending (the long-poll timed out — call again to keep blocking).
            result (Any | Unset): The servicer's return value when outcome=='ok'; null otherwise.
            error (AwaitResponseErrorType0 | None | Unset): On outcome 'errored'/'cancelled', the {kind, message, …} detail;
                null otherwise.
    """

    outcome: AwaitResponseOutcomeType0 | None | Unset = UNSET
    result: Any | Unset = UNSET
    error: AwaitResponseErrorType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.await_response_error_type_0 import AwaitResponseErrorType0

        outcome: None | str | Unset
        if isinstance(self.outcome, Unset):
            outcome = UNSET
        elif isinstance(self.outcome, AwaitResponseOutcomeType0):
            outcome = self.outcome.value
        else:
            outcome = self.outcome

        result = self.result

        error: dict[str, Any] | None | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        elif isinstance(self.error, AwaitResponseErrorType0):
            error = self.error.to_dict()
        else:
            error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if outcome is not UNSET:
            field_dict["outcome"] = outcome
        if result is not UNSET:
            field_dict["result"] = result
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.await_response_error_type_0 import AwaitResponseErrorType0

        d = dict(src_dict)

        def _parse_outcome(data: object) -> AwaitResponseOutcomeType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                outcome_type_0 = AwaitResponseOutcomeType0(data)

                return outcome_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AwaitResponseOutcomeType0 | None | Unset, data)

        outcome = _parse_outcome(d.pop("outcome", UNSET))

        result = d.pop("result", UNSET)

        def _parse_error(data: object) -> AwaitResponseErrorType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                error_type_0 = AwaitResponseErrorType0.from_dict(data)

                return error_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AwaitResponseErrorType0 | None | Unset, data)

        error = _parse_error(d.pop("error", UNSET))

        await_response = cls(
            outcome=outcome,
            result=result,
            error=error,
        )

        await_response.additional_properties = d
        return await_response

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
