from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.workflow_input_schema_type_0 import WorkflowInputSchemaType0
    from ..models.workflow_output_schema_type_0 import WorkflowOutputSchemaType0


T = TypeVar("T", bound="Workflow")


@_attrs_define
class Workflow:
    """An immutable, versioned workflow definition.

    Attributes:
        id (str):
        account_id (str):
        name (str):
        version (int):
        script (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        input_schema (None | Unset | WorkflowInputSchemaType0):
        output_schema (None | Unset | WorkflowOutputSchemaType0):
        description (None | str | Unset):
    """

    id: str
    account_id: str
    name: str
    version: int
    script: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    input_schema: None | Unset | WorkflowInputSchemaType0 = UNSET
    output_schema: None | Unset | WorkflowOutputSchemaType0 = UNSET
    description: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.workflow_input_schema_type_0 import WorkflowInputSchemaType0
        from ..models.workflow_output_schema_type_0 import WorkflowOutputSchemaType0

        id = self.id

        account_id = self.account_id

        name = self.name

        version = self.version

        script = self.script

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        input_schema: dict[str, Any] | None | Unset
        if isinstance(self.input_schema, Unset):
            input_schema = UNSET
        elif isinstance(self.input_schema, WorkflowInputSchemaType0):
            input_schema = self.input_schema.to_dict()
        else:
            input_schema = self.input_schema

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, WorkflowOutputSchemaType0):
            output_schema = self.output_schema.to_dict()
        else:
            output_schema = self.output_schema

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "account_id": account_id,
                "name": name,
                "version": version,
                "script": script,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if input_schema is not UNSET:
            field_dict["input_schema"] = input_schema
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema
        if description is not UNSET:
            field_dict["description"] = description

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.workflow_input_schema_type_0 import WorkflowInputSchemaType0
        from ..models.workflow_output_schema_type_0 import WorkflowOutputSchemaType0

        d = dict(src_dict)
        id = d.pop("id")

        account_id = d.pop("account_id")

        name = d.pop("name")

        version = d.pop("version")

        script = d.pop("script")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_input_schema(
            data: object,
        ) -> None | Unset | WorkflowInputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_schema_type_0 = WorkflowInputSchemaType0.from_dict(data)

                return input_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowInputSchemaType0, data)

        input_schema = _parse_input_schema(d.pop("input_schema", UNSET))

        def _parse_output_schema(
            data: object,
        ) -> None | Unset | WorkflowOutputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = WorkflowOutputSchemaType0.from_dict(data)

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowOutputSchemaType0, data)

        output_schema = _parse_output_schema(d.pop("output_schema", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        workflow = cls(
            id=id,
            account_id=account_id,
            name=name,
            version=version,
            script=script,
            created_at=created_at,
            updated_at=updated_at,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
        )

        workflow.additional_properties = d
        return workflow

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
