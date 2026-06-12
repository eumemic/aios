from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec
    from ..models.workflow_create_input_schema_type_0 import (
        WorkflowCreateInputSchemaType0,
    )
    from ..models.workflow_create_output_schema_type_0 import (
        WorkflowCreateOutputSchemaType0,
    )


T = TypeVar("T", bound="WorkflowCreate")


@_attrs_define
class WorkflowCreate:
    """Request body for ``POST /v1/workflows`` — a new workflow definition at v1.

    Attributes:
        name (str):
        script (str): Workflow script contract:
            - Entry point: define `async def main(input)`. A run's output is the value returned by
              `main`.
            - Injected capability API, available without imports:
              - `agent(agent_id, input, output_schema=None)`: invoke an agent and await its result.
              - `tool(name, input)`: invoke a declared tool; tool errors are returned, not raised.
              - `gate()`: suspend until an external resume delivers a value.
              - `budget()`: read this run's shared child-spend budget, or None when unset.
              - `parallel(thunks)`: run zero-argument callables concurrently (for example,
                `lambda: agent(...)`). A failed agent branch yields `None` at the barrier instead
                of raising. Fan-out width is capped by `MAX_PARALLEL_FANOUT` (currently 1000).
              - `pipeline(items, *stages)`: run each item through staged transforms concurrently.
              - `log(msg)`: record progress on the run journal.
            - Shell execution: `tool('bash', {"command": str, "timeout_s": float | None})` runs the
              command in a per-run sandbox (provisioned lazily on first use, in the run's
              environment). `bash` must be a member of the workflow's declared tools or the call
              resolves to a `{"error": ...}` value. Result: `{exit_code, stdout, stderr, timed_out,
              truncated}` — a nonzero exit or in-command timeout is a successful result to branch
              on, not an error.
            - Crash semantics: at-least-once at the call boundary. A capability call interrupted by
              a crash re-runs on resume; completed calls never re-run. The sandbox filesystem is
              ephemeral scratch — write re-run-tolerant commands (e.g. `rm -rf dir && git clone ...`).
            - Irreversible external effects (a POST that charges, sends, or publishes): the bash
              environment exposes `$AIOS_IDEMPOTENCY_KEY` (stable across crash re-runs of the same
              call, distinct per call) — pass it to the external service as an idempotency key so
              the service drops a re-fired duplicate, or knowingly accept at-least-once.
            - Partition rule: put re-run-tolerant mechanical work in `tool('bash')`; put work whose
              uncertain completion needs judgment to resolve inside `agent(...)`.
            - Environment: the SCRIPT runs in a deterministic, credential-free, isolated child
              process — imports restricted to a curated stdlib allowlist, no network or filesystem
              access from script code itself; all effects go through the capability API. The
              `tool('bash')` sandbox is different: it has a filesystem (ephemeral scratch) and
              network egress per the run's ENVIRONMENT network policy (Unrestricted, or Limited to
              the environment's allowed hosts) — commands can curl, clone, and install within that
              policy.

            Minimal example:
            ```python
            async def main(input):
                result = await agent(
                    input["agent_id"],
                    {"task": input["task"]},
                    None,
                )
                return result
            ```
        input_schema (None | Unset | WorkflowCreateInputSchemaType0):
        output_schema (None | Unset | WorkflowCreateOutputSchemaType0):
        description (None | str | Unset):
        tools (list[ToolSpec] | Unset):
        mcp_servers (list[McpServerSpec] | Unset):
        http_servers (list[HttpServerSpec] | Unset):
    """

    name: str
    script: str
    input_schema: None | Unset | WorkflowCreateInputSchemaType0 = UNSET
    output_schema: None | Unset | WorkflowCreateOutputSchemaType0 = UNSET
    description: None | str | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.workflow_create_input_schema_type_0 import (
            WorkflowCreateInputSchemaType0,
        )
        from ..models.workflow_create_output_schema_type_0 import (
            WorkflowCreateOutputSchemaType0,
        )

        name = self.name

        script = self.script

        input_schema: dict[str, Any] | None | Unset
        if isinstance(self.input_schema, Unset):
            input_schema = UNSET
        elif isinstance(self.input_schema, WorkflowCreateInputSchemaType0):
            input_schema = self.input_schema.to_dict()
        else:
            input_schema = self.input_schema

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, WorkflowCreateOutputSchemaType0):
            output_schema = self.output_schema.to_dict()
        else:
            output_schema = self.output_schema

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        mcp_servers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.mcp_servers, Unset):
            mcp_servers = []
            for mcp_servers_item_data in self.mcp_servers:
                mcp_servers_item = mcp_servers_item_data.to_dict()
                mcp_servers.append(mcp_servers_item)

        http_servers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.http_servers, Unset):
            http_servers = []
            for http_servers_item_data in self.http_servers:
                http_servers_item = http_servers_item_data.to_dict()
                http_servers.append(http_servers_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "script": script,
            }
        )
        if input_schema is not UNSET:
            field_dict["input_schema"] = input_schema
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema
        if description is not UNSET:
            field_dict["description"] = description
        if tools is not UNSET:
            field_dict["tools"] = tools
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if http_servers is not UNSET:
            field_dict["http_servers"] = http_servers

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec
        from ..models.workflow_create_input_schema_type_0 import (
            WorkflowCreateInputSchemaType0,
        )
        from ..models.workflow_create_output_schema_type_0 import (
            WorkflowCreateOutputSchemaType0,
        )

        d = dict(src_dict)
        name = d.pop("name")

        script = d.pop("script")

        def _parse_input_schema(
            data: object,
        ) -> None | Unset | WorkflowCreateInputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_schema_type_0 = WorkflowCreateInputSchemaType0.from_dict(data)

                return input_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowCreateInputSchemaType0, data)

        input_schema = _parse_input_schema(d.pop("input_schema", UNSET))

        def _parse_output_schema(
            data: object,
        ) -> None | Unset | WorkflowCreateOutputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = WorkflowCreateOutputSchemaType0.from_dict(data)

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowCreateOutputSchemaType0, data)

        output_schema = _parse_output_schema(d.pop("output_schema", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        _tools = d.pop("tools", UNSET)
        tools: list[ToolSpec] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = ToolSpec.from_dict(tools_item_data)

                tools.append(tools_item)

        _mcp_servers = d.pop("mcp_servers", UNSET)
        mcp_servers: list[McpServerSpec] | Unset = UNSET
        if _mcp_servers is not UNSET:
            mcp_servers = []
            for mcp_servers_item_data in _mcp_servers:
                mcp_servers_item = McpServerSpec.from_dict(mcp_servers_item_data)

                mcp_servers.append(mcp_servers_item)

        _http_servers = d.pop("http_servers", UNSET)
        http_servers: list[HttpServerSpec] | Unset = UNSET
        if _http_servers is not UNSET:
            http_servers = []
            for http_servers_item_data in _http_servers:
                http_servers_item = HttpServerSpec.from_dict(http_servers_item_data)

                http_servers.append(http_servers_item)

        workflow_create = cls(
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
        )

        return workflow_create
