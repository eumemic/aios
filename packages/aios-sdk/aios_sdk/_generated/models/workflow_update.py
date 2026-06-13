from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec
    from ..models.workflow_update_input_schema_type_0 import (
        WorkflowUpdateInputSchemaType0,
    )
    from ..models.workflow_update_output_schema_type_0 import (
        WorkflowUpdateOutputSchemaType0,
    )


T = TypeVar("T", bound="WorkflowUpdate")


@_attrs_define
class WorkflowUpdate:
    """Request body for ``PUT /v1/workflows/{id}`` — update in place, bumping ``version``.

    ``version`` is the optimistic-concurrency token: it must match the workflow's
    current version or the update 409s (re-fetch and retry). Omitted fields are
    preserved — nullable fields (``input_schema``/``output_schema``/``description``)
    can therefore be replaced but never cleared back to null, as on ``AgentUpdate``.
    An identical update is a no-op (no bump). There is no version-snapshot table —
    a run pins ``script`` + the declared surface onto itself at launch, so in-flight
    runs never observe an update. (The ``AgentUpdate`` shape, minus history.)

        Attributes:
            version (int):
            name (None | str | Unset):
            script (None | str | Unset): Workflow script contract:
                - Entry point: define `async def main(input)`. A run's output is the value returned by
                  `main`.
                - Injected capability API, available without imports:
                  - `agent(input, *, agent_id=None, output_schema=None, model=None, label=None)`: invoke a generic or named
                agent and await its result.
                  - `tool(name, input)`: invoke a declared tool; tool errors are returned, not raised.
                  - `gate()`: suspend until an external resume delivers a value.
                  - `budget()`: read this run's shared child-spend budget, or None when unset.
                  - `parallel(thunks)`: run zero-argument callables concurrently (for example,
                    `lambda: agent(...)`). A failed agent branch yields `None` at the barrier instead
                    of raising. Fan-out width is capped by `MAX_PARALLEL_FANOUT` (currently 1000).
                  - `pipeline(items, *stages)`: run each item through staged transforms concurrently.
                  - `log(msg)`: record progress on the run journal.
                  - `phase(label)`: record a phase marker on the run journal.
                - Shell execution: `tool('bash', {"command": str, "timeout_seconds": float | None})` runs the
                  command in a per-run sandbox (provisioned lazily on first use, in the run's
                  environment). `bash` must be a member of the workflow's declared tools or the call
                  resolves to a `{"error": ...}` value. Result: `{exit_code, stdout, stderr, timed_out,
                  truncated}` — a nonzero exit or in-command timeout is a successful result to branch
                  on, not an error.
                - Crash semantics: at-least-once at the call boundary. A capability call interrupted by
                  a crash re-runs on resume; completed calls never re-run. The sandbox filesystem is
                  ephemeral scratch — write re-run-tolerant commands (e.g. `rm -rf dir && git clone ...`).
                - Irreversible external effects (a POST that charges, sends, or publishes): a per-call
                  idempotency token (stable across crash re-runs of the same call, distinct per call) is
                  available so you can have the external service drop a re-fired duplicate, or knowingly
                  accept at-least-once. Both deliveries opt in the same way with the same `$AIOS_IDEMPOTENCY_KEY`
                  ergonomic: in `tool('bash')` the environment exposes `$AIOS_IDEMPOTENCY_KEY`; in
                  `tool('http_request')` pass the sentinel string `"$AIOS_IDEMPOTENCY_KEY"` as an
                  `Idempotency-Key` header value and the worker substitutes the real token before dispatch.
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
                        {"task": input["task"]},
                        agent_id=input["agent_id"],
                    )
                    return result
                ```
            input_schema (None | Unset | WorkflowUpdateInputSchemaType0):
            output_schema (None | Unset | WorkflowUpdateOutputSchemaType0):
            description (None | str | Unset):
            tools (list[ToolSpec] | None | Unset):
            mcp_servers (list[McpServerSpec] | None | Unset):
            http_servers (list[HttpServerSpec] | None | Unset):
    """

    version: int
    name: None | str | Unset = UNSET
    script: None | str | Unset = UNSET
    input_schema: None | Unset | WorkflowUpdateInputSchemaType0 = UNSET
    output_schema: None | Unset | WorkflowUpdateOutputSchemaType0 = UNSET
    description: None | str | Unset = UNSET
    tools: list[ToolSpec] | None | Unset = UNSET
    mcp_servers: list[McpServerSpec] | None | Unset = UNSET
    http_servers: list[HttpServerSpec] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.workflow_update_input_schema_type_0 import (
            WorkflowUpdateInputSchemaType0,
        )
        from ..models.workflow_update_output_schema_type_0 import (
            WorkflowUpdateOutputSchemaType0,
        )

        version = self.version

        name: None | str | Unset
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        script: None | str | Unset
        if isinstance(self.script, Unset):
            script = UNSET
        else:
            script = self.script

        input_schema: dict[str, Any] | None | Unset
        if isinstance(self.input_schema, Unset):
            input_schema = UNSET
        elif isinstance(self.input_schema, WorkflowUpdateInputSchemaType0):
            input_schema = self.input_schema.to_dict()
        else:
            input_schema = self.input_schema

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, WorkflowUpdateOutputSchemaType0):
            output_schema = self.output_schema.to_dict()
        else:
            output_schema = self.output_schema

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        tools: list[dict[str, Any]] | None | Unset
        if isinstance(self.tools, Unset):
            tools = UNSET
        elif isinstance(self.tools, list):
            tools = []
            for tools_type_0_item_data in self.tools:
                tools_type_0_item = tools_type_0_item_data.to_dict()
                tools.append(tools_type_0_item)

        else:
            tools = self.tools

        mcp_servers: list[dict[str, Any]] | None | Unset
        if isinstance(self.mcp_servers, Unset):
            mcp_servers = UNSET
        elif isinstance(self.mcp_servers, list):
            mcp_servers = []
            for mcp_servers_type_0_item_data in self.mcp_servers:
                mcp_servers_type_0_item = mcp_servers_type_0_item_data.to_dict()
                mcp_servers.append(mcp_servers_type_0_item)

        else:
            mcp_servers = self.mcp_servers

        http_servers: list[dict[str, Any]] | None | Unset
        if isinstance(self.http_servers, Unset):
            http_servers = UNSET
        elif isinstance(self.http_servers, list):
            http_servers = []
            for http_servers_type_0_item_data in self.http_servers:
                http_servers_type_0_item = http_servers_type_0_item_data.to_dict()
                http_servers.append(http_servers_type_0_item)

        else:
            http_servers = self.http_servers

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "version": version,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if script is not UNSET:
            field_dict["script"] = script
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
        from ..models.workflow_update_input_schema_type_0 import (
            WorkflowUpdateInputSchemaType0,
        )
        from ..models.workflow_update_output_schema_type_0 import (
            WorkflowUpdateOutputSchemaType0,
        )

        d = dict(src_dict)
        version = d.pop("version")

        def _parse_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_script(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        script = _parse_script(d.pop("script", UNSET))

        def _parse_input_schema(
            data: object,
        ) -> None | Unset | WorkflowUpdateInputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_schema_type_0 = WorkflowUpdateInputSchemaType0.from_dict(data)

                return input_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowUpdateInputSchemaType0, data)

        input_schema = _parse_input_schema(d.pop("input_schema", UNSET))

        def _parse_output_schema(
            data: object,
        ) -> None | Unset | WorkflowUpdateOutputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = WorkflowUpdateOutputSchemaType0.from_dict(data)

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowUpdateOutputSchemaType0, data)

        output_schema = _parse_output_schema(d.pop("output_schema", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_tools(data: object) -> list[ToolSpec] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = []
                _tools_type_0 = data
                for tools_type_0_item_data in _tools_type_0:
                    tools_type_0_item = ToolSpec.from_dict(tools_type_0_item_data)

                    tools_type_0.append(tools_type_0_item)

                return tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ToolSpec] | None | Unset, data)

        tools = _parse_tools(d.pop("tools", UNSET))

        def _parse_mcp_servers(data: object) -> list[McpServerSpec] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                mcp_servers_type_0 = []
                _mcp_servers_type_0 = data
                for mcp_servers_type_0_item_data in _mcp_servers_type_0:
                    mcp_servers_type_0_item = McpServerSpec.from_dict(
                        mcp_servers_type_0_item_data
                    )

                    mcp_servers_type_0.append(mcp_servers_type_0_item)

                return mcp_servers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[McpServerSpec] | None | Unset, data)

        mcp_servers = _parse_mcp_servers(d.pop("mcp_servers", UNSET))

        def _parse_http_servers(data: object) -> list[HttpServerSpec] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                http_servers_type_0 = []
                _http_servers_type_0 = data
                for http_servers_type_0_item_data in _http_servers_type_0:
                    http_servers_type_0_item = HttpServerSpec.from_dict(
                        http_servers_type_0_item_data
                    )

                    http_servers_type_0.append(http_servers_type_0_item)

                return http_servers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[HttpServerSpec] | None | Unset, data)

        http_servers = _parse_http_servers(d.pop("http_servers", UNSET))

        workflow_update = cls(
            version=version,
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
        )

        return workflow_update
