from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="WfRunUsage")


@_attrs_define
class WfRunUsage:
    """Per-run cost / token / iteration / wall-clock — the machine-observer's substrate.

    The read-path projection of a run's actual spend (#1324). The numbers are summed
    over the run's direct child sessions via ``run_children_usage`` — the SAME source
    ``step.py``'s ``budget()`` builtin consumes — so a run's ``budget_usd`` *ceiling*
    (on ``WfRun``) and its realized ``cost_microusd`` *spend* (here) are finally both
    legible from the read path.

    EVERY field is ``int | None``, and absence is reported as **explicit null**, never
    a silent ``0`` or an omitted key (cf. the ``vault_ids:null`` read-path disease this
    must not inherit — see the substrate-different-verdict invariant). The observer
    reads null as *cannot-determine* and fails loud, NOT as "zero spend":

    * ``cost_microusd`` / ``*_tokens`` — summed over the run's child sessions. A run
      with no children sums to ``0`` (a real, observed zero — distinct from null).
    * ``iteration_count`` — the run's wake/step count. The host keeps **no** per-run
      iteration counter on any substrate today, so this is reported as ``None``
      (cannot-determine) rather than fabricated from an unrelated proxy. Reserved for
      when a real counter lands; surfaced now so the observer's contract is stable.
    * ``wall_clock_ms`` — wall-clock span ``updated_at - created_at`` in milliseconds,
      reported ONLY for a TERMINAL run (``updated_at`` is its completion instant). A
      still-running run's ``updated_at`` is a moving "last touched" stamp, not an end,
      so it is reported as ``None`` rather than a misleading partial span.

        Attributes:
            cost_microusd (int | None | Unset):
            input_tokens (int | None | Unset):
            output_tokens (int | None | Unset):
            cache_read_input_tokens (int | None | Unset):
            cache_creation_input_tokens (int | None | Unset):
            iteration_count (int | None | Unset):
            wall_clock_ms (int | None | Unset):
    """

    cost_microusd: int | None | Unset = UNSET
    input_tokens: int | None | Unset = UNSET
    output_tokens: int | None | Unset = UNSET
    cache_read_input_tokens: int | None | Unset = UNSET
    cache_creation_input_tokens: int | None | Unset = UNSET
    iteration_count: int | None | Unset = UNSET
    wall_clock_ms: int | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_microusd: int | None | Unset
        if isinstance(self.cost_microusd, Unset):
            cost_microusd = UNSET
        else:
            cost_microusd = self.cost_microusd

        input_tokens: int | None | Unset
        if isinstance(self.input_tokens, Unset):
            input_tokens = UNSET
        else:
            input_tokens = self.input_tokens

        output_tokens: int | None | Unset
        if isinstance(self.output_tokens, Unset):
            output_tokens = UNSET
        else:
            output_tokens = self.output_tokens

        cache_read_input_tokens: int | None | Unset
        if isinstance(self.cache_read_input_tokens, Unset):
            cache_read_input_tokens = UNSET
        else:
            cache_read_input_tokens = self.cache_read_input_tokens

        cache_creation_input_tokens: int | None | Unset
        if isinstance(self.cache_creation_input_tokens, Unset):
            cache_creation_input_tokens = UNSET
        else:
            cache_creation_input_tokens = self.cache_creation_input_tokens

        iteration_count: int | None | Unset
        if isinstance(self.iteration_count, Unset):
            iteration_count = UNSET
        else:
            iteration_count = self.iteration_count

        wall_clock_ms: int | None | Unset
        if isinstance(self.wall_clock_ms, Unset):
            wall_clock_ms = UNSET
        else:
            wall_clock_ms = self.wall_clock_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if cost_microusd is not UNSET:
            field_dict["cost_microusd"] = cost_microusd
        if input_tokens is not UNSET:
            field_dict["input_tokens"] = input_tokens
        if output_tokens is not UNSET:
            field_dict["output_tokens"] = output_tokens
        if cache_read_input_tokens is not UNSET:
            field_dict["cache_read_input_tokens"] = cache_read_input_tokens
        if cache_creation_input_tokens is not UNSET:
            field_dict["cache_creation_input_tokens"] = cache_creation_input_tokens
        if iteration_count is not UNSET:
            field_dict["iteration_count"] = iteration_count
        if wall_clock_ms is not UNSET:
            field_dict["wall_clock_ms"] = wall_clock_ms

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_cost_microusd(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        cost_microusd = _parse_cost_microusd(d.pop("cost_microusd", UNSET))

        def _parse_input_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        input_tokens = _parse_input_tokens(d.pop("input_tokens", UNSET))

        def _parse_output_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        output_tokens = _parse_output_tokens(d.pop("output_tokens", UNSET))

        def _parse_cache_read_input_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        cache_read_input_tokens = _parse_cache_read_input_tokens(
            d.pop("cache_read_input_tokens", UNSET)
        )

        def _parse_cache_creation_input_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        cache_creation_input_tokens = _parse_cache_creation_input_tokens(
            d.pop("cache_creation_input_tokens", UNSET)
        )

        def _parse_iteration_count(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        iteration_count = _parse_iteration_count(d.pop("iteration_count", UNSET))

        def _parse_wall_clock_ms(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        wall_clock_ms = _parse_wall_clock_ms(d.pop("wall_clock_ms", UNSET))

        wf_run_usage = cls(
            cost_microusd=cost_microusd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            iteration_count=iteration_count,
            wall_clock_ms=wall_clock_ms,
        )

        wf_run_usage.additional_properties = d
        return wf_run_usage

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
