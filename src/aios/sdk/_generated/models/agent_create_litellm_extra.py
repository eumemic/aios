from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="AgentCreateLitellmExtra")


@_attrs_define
class AgentCreateLitellmExtra:
    """Provider-specific LiteLLM kwargs merged into every model request for this agent.  Common shapes: OpenRouter
    ``extra_body.provider.order`` for provider pinning, Anthropic ``thinking``, OpenAI ``reasoning_effort``, raw
    sampling knobs (``temperature``, ``max_tokens``), ``api_base`` for self-hosted inference.  Validated by LiteLLM /
    the provider; bad kwargs surface as tool-path errors the model sees.  Security: ``api_base`` redirects the model
    call — treat operator-set agents as trusted and don't accept this field from untrusted principals.

    """

    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        agent_create_litellm_extra = cls()

        agent_create_litellm_extra.additional_properties = d
        return agent_create_litellm_extra

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
