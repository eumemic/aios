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
              - `agent(input, *, agent_id=None, output_schema=None, model=None, label=None)`: invoke a generic or named
            agent and await its result.
              - `invoke_workflow(workflow_id, input, *, output_schema=None, label=None)`: invoke another workflow as a sub-
            run and await its result (the run dual of `agent`). The sub-run runs under this run's surface intersected with
            the target's; a failed or gone sub-run raises like a failed `agent`.
              - `tool(name, input)`: invoke a declared tool; tool errors are returned, not raised.
              - `call_llm(request)`: run one raw inference turn and await the assistant turn. `request` carries `model`
            (omit to use the run's default child model; a `workflow:` target is rejected), `messages` (required), optional
            `tools` (schemas OFFERED — the model may request a call, but call_llm never runs it), and optional `params`
            (provider knobs). The result is `{"content", "tool_calls", "finish_reason", "usage", "cost", "message"}`, or
            `{"error": ...}` — a model error is returned, not raised. Its cost is metered against this run's `budget_usd`
            ceiling, so a budget-exhausted run refuses further `call_llm`. Use it to route/judge/fact-check around
            inference; use `agent(...)` when you want the tool calls executed.
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
        input_schema (None | Unset | WorkflowCreateInputSchemaType0):
        output_schema (None | Unset | WorkflowCreateOutputSchemaType0):
        output_model (None | str | Unset):
        description (None | str | Unset):
        tools (list[ToolSpec] | Unset):
        mcp_servers (list[McpServerSpec] | Unset):
        http_servers (list[HttpServerSpec | str] | Unset):
    """

    name: str
    script: str
    input_schema: None | Unset | WorkflowCreateInputSchemaType0 = UNSET
    output_schema: None | Unset | WorkflowCreateOutputSchemaType0 = UNSET
    output_model: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec | str] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.http_server_spec import HttpServerSpec
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

        output_model: None | str | Unset
        if isinstance(self.output_model, Unset):
            output_model = UNSET
        else:
            output_model = self.output_model

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

        http_servers: list[dict[str, Any] | str] | Unset = UNSET
        if not isinstance(self.http_servers, Unset):
            http_servers = []
            for http_servers_item_data in self.http_servers:
                http_servers_item: dict[str, Any] | str
                if isinstance(http_servers_item_data, HttpServerSpec):
                    http_servers_item = http_servers_item_data.to_dict()
                else:
                    http_servers_item = http_servers_item_data
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
        if output_model is not UNSET:
            field_dict["output_model"] = output_model
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

        def _parse_output_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        output_model = _parse_output_model(d.pop("output_model", UNSET))

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
        http_servers: list[HttpServerSpec | str] | Unset = UNSET
        if _http_servers is not UNSET:
            http_servers = []
            for http_servers_item_data in _http_servers:

                def _parse_http_servers_item(data: object) -> HttpServerSpec | str:
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        http_servers_item_type_1 = HttpServerSpec.from_dict(data)

                        return http_servers_item_type_1
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    return cast(HttpServerSpec | str, data)

                http_servers_item = _parse_http_servers_item(http_servers_item_data)

                http_servers.append(http_servers_item)

        workflow_create = cls(
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            output_model=output_model,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
        )

        return workflow_create
