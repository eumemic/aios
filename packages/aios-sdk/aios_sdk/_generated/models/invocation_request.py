from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.invocation_request_target_kind import InvocationRequestTargetKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.invocation_request_output_schema_type_0 import (
        InvocationRequestOutputSchemaType0,
    )


T = TypeVar("T", bound="InvocationRequest")


@_attrs_define
class InvocationRequest:
    """Request body for ``POST /v1/invocations`` — the API caller's request-writer.

    ``target`` is an ``agent_id | workflow_id | session_id`` and ``target_kind``
    discriminates it:

    * ``agent``     — create a **session** servicer and inject a channel-less
      request into it (the API analog of ``invoke_agent``).
    * ``workflow``  — create a **run** servicer of the workflow.
    * ``session``   — invoke an **existing** same-account session by id (the API
      analog of #1127's ``invoke(session_id)``). No ``environment_id`` applies —
      the session already exists.

    ``output_schema`` is the per-request JSON Schema the response ``value`` must
    satisfy; it rides the edge (``metadata.request.output_schema``), coexisting
    with any definition-level schema. ``environment_id`` is ownership-checked
    against the caller's account on the ``agent`` / ``workflow`` create-paths
    (the per-field containment clamp is #1130's deliverable).

        Attributes:
            target_kind (InvocationRequestTargetKind):
            target (str): An agent_id / workflow_id / session_id, discriminated by ``target_kind``.
            input_ (Any | Unset): The request payload delivered to the servicer (arbitrary JSON or a string).
            output_schema (InvocationRequestOutputSchemaType0 | None | Unset): Optional JSON Schema the response ``value``
                must satisfy. Rides the request edge, per-request — coexists with any definition-level schema.
            environment_id (None | str | Unset): Environment to bind a created servicer to (agent → session, workflow →
                run). Ownership-checked against the caller's account. Inapplicable for ``target_kind=session`` (the session
                already exists).
    """

    target_kind: InvocationRequestTargetKind
    target: str
    input_: Any | Unset = UNSET
    output_schema: InvocationRequestOutputSchemaType0 | None | Unset = UNSET
    environment_id: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.invocation_request_output_schema_type_0 import (
            InvocationRequestOutputSchemaType0,
        )

        target_kind = self.target_kind.value

        target = self.target

        input_ = self.input_

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, InvocationRequestOutputSchemaType0):
            output_schema = self.output_schema.to_dict()
        else:
            output_schema = self.output_schema

        environment_id: None | str | Unset
        if isinstance(self.environment_id, Unset):
            environment_id = UNSET
        else:
            environment_id = self.environment_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "target_kind": target_kind,
                "target": target,
            }
        )
        if input_ is not UNSET:
            field_dict["input"] = input_
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema
        if environment_id is not UNSET:
            field_dict["environment_id"] = environment_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.invocation_request_output_schema_type_0 import (
            InvocationRequestOutputSchemaType0,
        )

        d = dict(src_dict)
        target_kind = InvocationRequestTargetKind(d.pop("target_kind"))

        target = d.pop("target")

        input_ = d.pop("input", UNSET)

        def _parse_output_schema(
            data: object,
        ) -> InvocationRequestOutputSchemaType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = InvocationRequestOutputSchemaType0.from_dict(
                    data
                )

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(InvocationRequestOutputSchemaType0 | None | Unset, data)

        output_schema = _parse_output_schema(d.pop("output_schema", UNSET))

        def _parse_environment_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        environment_id = _parse_environment_id(d.pop("environment_id", UNSET))

        invocation_request = cls(
            target_kind=target_kind,
            target=target,
            input_=input_,
            output_schema=output_schema,
            environment_id=environment_id,
        )

        return invocation_request
