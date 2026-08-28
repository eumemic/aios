"""MCP tool schemas — sanitize inputSchemas, build OpenAI function-tool envelopes."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from mcp.types import Tool

# Anthropic (and OpenAI) function-tool names: ``^[a-zA-Z0-9_-]{1,64}$``.
# MCP SEP-986 is more permissive (``.`` / ``/``) and does not enforce 64; the
# provider 400s. Qualified names are ``mcp__<server>__<tool>``.
PROVIDER_TOOL_NAME_MAX = 64
_PROVIDER_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_UNSAFE_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")

# Envelope key (sibling of ``function``, never inside it) mapping an advertised
# name back to the MCP server + raw tool. Stripped before the provider call.
MCP_ORIGIN_KEY = "_mcp_origin"

# Envelope key (sibling of ``function``, never inside it) carrying the raw MCP
# ``annotations`` dict (``readOnlyHint``/``destructiveHint``/etc, per the MCP
# spec) as reported at discovery time. Internal-only, like ``MCP_ORIGIN_KEY`` —
# stripped before the provider call since Anthropic/OpenAI don't accept it.
# Consulted by the auto-allow-readonly composer logic (#2270) so permissioning
# doesn't need a second raw-discovery round trip.
MCP_ANNOTATIONS_KEY = "_mcp_annotations"


# Structural-walk keyword sets, mirroring ``aios.tools.schema_diet``'s
# ``_walk_schemas`` (see its docstring for why a blind ``dict`` walk is wrong:
# inside ``properties`` the KEYS are field names, never schema keywords).
#: Keywords whose value is a NAME → schema mapping.
_SCHEMA_MAPS = ("properties", "$defs", "patternProperties", "definitions")
#: Keywords whose value is a list of schemas.
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
#: Keywords whose value is a single schema.
_SCHEMA_VALUES = ("items", "not", "contains", "additionalProperties", "propertyNames")


def sanitize_mcp_schema(node: Any) -> Any:
    """Rewrite an untrusted schema into the subset litellm + providers tolerate.

    Three repairs, applied along a **structural** walk that descends only
    through known schema-composition keywords (``properties`` values,
    ``items``, ``$defs``, ``anyOf``/``oneOf``/``allOf`` members,
    ``additionalProperties``/``propertyNames`` when dicts, …). The distinction
    is load-bearing: a properties MAP whose keys happen to include ``properties``
    or ``type`` (a tool parameter literally named after a keyword — Notion's
    ``notion-create-pages`` has a page ``properties`` field) must never be
    mistaken for a schema node, or its legal keyword values (``"type":
    "object"``) get mangled into ``{}`` and Anthropic 400s the whole request
    (draft 2020-12 validation).

    * Drop the ``type`` keyword next to ``anyOf``/``oneOf`` (the union carries
      the real shape). Only the JSON Schema ``type`` **keyword** — whose value is
      a type name (``"string"``) or a list of them (``["string", "null"]``) — is
      redundant beside a union. A property literally **named** ``type`` (its
      value is a sub-schema ``dict``) inside a ``properties`` map is a parameter,
      not a keyword, and must be preserved even when a sibling property is named
      ``anyOf``/``oneOf``; aios sanitizes untrusted third-party tool schemas, so
      dropping it would silently corrupt a valid tool and make the model call it
      wrong.

    * On ``type == "array"``, ensure ``items`` is a dict: litellm's
      ``token_counter`` → ``_format_type`` dereferences ``props['items']``
      unconditionally (``KeyError: 'items'`` when missing — the #2294 incident
      class) and recurses with ``.get`` (``AttributeError`` on draft-4 tuple-form
      ``items: [...]`` or boolean ``items``). OpenAI also rejects arrays without
      ``items``. ``{}`` is the "anything" schema, so the repair only loosens.

    * Coerce non-dict ``properties`` values (boolean schemas like ``"foo": true``)
      to ``{}`` — litellm's ``_format_object_parameters`` calls ``.get`` on every
      property value. Applied only to a genuine ``properties`` map on a schema
      node reached through the structural walk.
    """
    if isinstance(node, dict):
        return _sanitize_schema(node)
    if isinstance(node, list):
        return [sanitize_mcp_schema(item) for item in node]
    return node


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Repair one schema node and recurse structurally into its subschemas."""
    has_union = "anyOf" in schema or "oneOf" in schema
    drop_type = has_union and isinstance(schema.get("type"), (str, list))
    cleaned = {
        key: _sanitize_keyword_value(key, value)
        for key, value in schema.items()
        if not (drop_type and key == "type")
    }
    if cleaned.get("type") == "array" and not isinstance(cleaned.get("items"), dict):
        cleaned["items"] = {}
    return cleaned


def _sanitize_keyword_value(key: str, value: Any) -> Any:
    """Recurse into ``value`` only when ``key`` is a schema-composition keyword.

    Anything else (``description``, ``default``, ``enum``, ``const``, unknown
    keywords) is data, not schema — deep-copied untouched so the sanitized
    result never aliases the caller's input.
    """
    if key in _SCHEMA_MAPS and isinstance(value, dict):
        coerce = key == "properties"
        return {
            name: _sanitize_schema(sub)
            if isinstance(sub, dict)
            else ({} if coerce else copy.deepcopy(sub))
            for name, sub in value.items()
        }
    if key in _SCHEMA_LISTS and isinstance(value, list):
        return [
            _sanitize_schema(arm) if isinstance(arm, dict) else copy.deepcopy(arm) for arm in value
        ]
    if key in _SCHEMA_VALUES and isinstance(value, dict):
        return _sanitize_schema(value)
    return copy.deepcopy(value)


def _sanitize_name_segment(raw: str) -> str:
    cleaned = _UNSAFE_NAME_CHARS.sub("_", raw).strip("_")
    return cleaned or "x"


def qualify_mcp_tool_name(server_name: str, tool_name: str) -> tuple[str, str, str]:
    """Fit ``mcp__<server>__<tool>`` to the provider name regex.

    Returns ``(advertised, origin_server, origin_tool)``. Dispatch must call the
    origin tool on the origin server; the advertised name is what the model sees.
    """
    server = _sanitize_name_segment(server_name)
    tool = _sanitize_name_segment(tool_name)
    advertised = f"mcp__{server}__{tool}"
    if len(advertised) <= PROVIDER_TOOL_NAME_MAX and _PROVIDER_TOOL_NAME_RE.match(advertised):
        return advertised, server_name, tool_name
    digest = hashlib.sha1(f"{server_name}\0{tool_name}".encode()).hexdigest()
    prefix = f"mcp__{server}__"
    # ``<truncated>_<hash6>`` must fit after the prefix.
    room = PROVIDER_TOOL_NAME_MAX - len(prefix) - 7
    if room >= 1:
        fitted = f"{prefix}{tool[:room]}_{digest[:6]}"
        return fitted, server_name, tool_name
    # Server segment itself leaves no room for a tool — hash the server too.
    server_h = hashlib.sha1(server_name.encode()).hexdigest()[:8]
    prefix = f"mcp__{server_h}__"
    room = max(PROVIDER_TOOL_NAME_MAX - len(prefix) - 7, 1)
    fitted = f"{prefix}{tool[:room]}_{digest[:6]}"
    return fitted[:PROVIDER_TOOL_NAME_MAX], server_name, tool_name


def make_function_tool(
    qualified_name: str,
    tool: Tool,
    *,
    origin_server: str | None = None,
    origin_tool: str | None = None,
) -> dict[str, Any]:
    """Build the envelope, applying :func:`sanitize_mcp_schema` to ``tool.inputSchema``.

    When the tool declares an ``outputSchema`` (MCP 2025-06-18 structured
    output), keep it on the *internal* envelope so result-shaping and tests
    can see it (#1493). :func:`sanitize_tools_for_provider` strips it before
    the LiteLLM call — Anthropic rejects the unknown field with a 400
    (``drop_params`` is forced off, so LiteLLM will not silently drop it).
    """
    function: dict[str, Any] = {
        "name": qualified_name,
        "description": tool.description or "",
        "parameters": sanitize_mcp_schema(tool.inputSchema),
        # Arbitrary MCP schemas are not guaranteed to fit providers' narrower
        # strict-function subset. Explicit false also prevents provider-side
        # auto-promotion from turning optional MCP properties into required ones.
        "strict": False,
    }
    output_schema = getattr(tool, "outputSchema", None)
    if output_schema is not None:
        function["outputSchema"] = sanitize_mcp_schema(output_schema)
    envelope: dict[str, Any] = {
        "type": "function",
        "function": function,
    }
    if origin_server and origin_tool:
        envelope[MCP_ORIGIN_KEY] = {"server": origin_server, "tool": origin_tool}
    annotations = getattr(tool, "annotations", None)
    if annotations is not None:
        dumped = annotations.model_dump(exclude_none=True)
        if dumped:
            envelope[MCP_ANNOTATIONS_KEY] = dumped
    return envelope


def _function_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _advertised_names_are_unique(tools: list[dict[str, Any]]) -> bool:
    seen: set[str] = set()
    for tool in tools:
        name = _function_name(tool)
        if name is None:
            continue
        if name in seen:
            return False
        seen.add(name)
    return True


def _origin_pair(tool: dict[str, Any]) -> tuple[str, str] | None:
    origin = tool.get(MCP_ORIGIN_KEY)
    if not isinstance(origin, dict):
        return None
    server = origin.get("server")
    name = origin.get("tool")
    if isinstance(server, str) and server and isinstance(name, str) and name:
        return server, name
    return None


def _fit_name_with_suffix(base: str, tag: str) -> str:
    tag = _sanitize_name_segment(tag)
    suffix = f"_{tag}"
    room = PROVIDER_TOOL_NAME_MAX - len(suffix)
    if room < 1:
        tag = hashlib.sha1(tag.encode()).hexdigest()[:6]
        suffix = f"_{tag}"
        room = PROVIDER_TOOL_NAME_MAX - len(suffix)
    stem = base[:room].rstrip("_") or "mcp"
    return f"{stem}{suffix}"[:PROVIDER_TOOL_NAME_MAX]


def _disambiguate_advertised_name(base: str, server: str, used: set[str], *, salt: int) -> str:
    """Fold a short server-derived suffix (then hash) into ``base`` until unique."""
    server_seg = _sanitize_name_segment(server)
    digest = hashlib.sha1(f"{server}\0{base}\0{salt}".encode()).hexdigest()
    tags: list[str] = []
    for raw in (server_seg[-8:], digest[:6], digest[:8], f"{digest[:4]}{salt}"):
        tag = _sanitize_name_segment(raw)
        if tag and tag not in tags:
            tags.append(tag)
    for tag in tags:
        candidate = _fit_name_with_suffix(base, tag)
        if candidate not in used and _PROVIDER_TOOL_NAME_RE.match(candidate):
            return candidate
    raise RuntimeError(f"could not uniquify tool name {base!r}")


def uniquify_advertised_tool_names(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first advertised ``function.name``; rewrite later collisions.

    Copy-on-write: returns the input list when names are already unique.
    Rewritten envelopes keep/gain ``_mcp_origin`` so dispatch can map the
    new advertised name back to the raw server + tool.
    """
    if _advertised_names_are_unique(tools):
        return tools
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        name = _function_name(tool)
        if name is None or name not in seen:
            if name is not None:
                seen.add(name)
            out.append(tool)
            continue
        item = copy.deepcopy(tool)
        function = item.get("function")
        if not isinstance(function, dict):
            out.append(item)
            continue
        origin = _origin_pair(item)
        server = origin[0] if origin else f"s{index}"
        new_name = _disambiguate_advertised_name(name, server, seen, salt=index)
        function["name"] = new_name
        if origin:
            item[MCP_ORIGIN_KEY] = {"server": origin[0], "tool": origin[1]}
        seen.add(new_name)
        out.append(item)
    return out


def _tool_needs_provider_sanitize(tool: dict[str, Any]) -> bool:
    if MCP_ORIGIN_KEY in tool or MCP_ANNOTATIONS_KEY in tool:
        return True
    function = tool.get("function")
    if not isinstance(function, dict):
        return False
    if "outputSchema" in function:
        return True
    name = function.get("name")
    return isinstance(name, str) and _PROVIDER_TOOL_NAME_RE.match(name) is None


def sanitize_tools_for_provider(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy-on-write: drop fields / names Anthropic (and OpenAI) 400 on.

    Returns the input list object when nothing needs changing so existing
    identity assertions on a clean toolset keep holding. Advertised
    ``function.name`` values on the returned list are unique.
    """
    needs_copy = any(_tool_needs_provider_sanitize(tool) for tool in tools)
    if not needs_copy and _advertised_names_are_unique(tools):
        return tools
    cleaned: list[dict[str, Any]] = []
    for tool in tools:
        item = copy.deepcopy(tool) if needs_copy else tool
        if needs_copy:
            item.pop(MCP_ORIGIN_KEY, None)
            item.pop(MCP_ANNOTATIONS_KEY, None)
            function = item.get("function")
            if isinstance(function, dict):
                function.pop("outputSchema", None)
                name = function.get("name")
                if isinstance(name, str) and _PROVIDER_TOOL_NAME_RE.match(name) is None:
                    function["name"] = _sanitize_name_segment(name)[:PROVIDER_TOOL_NAME_MAX]
        cleaned.append(item)
    return uniquify_advertised_tool_names(cleaned)


def mcp_origin_for(
    qualified_name: str, tools: list[dict[str, Any]] | None
) -> tuple[str, str] | None:
    """Look up the raw ``(server, tool)`` for an advertised name, if stored."""
    for tool in tools or []:
        function = tool.get("function") or {}
        if function.get("name") != qualified_name:
            continue
        origin = tool.get(MCP_ORIGIN_KEY)
        if not isinstance(origin, dict):
            return None
        server = origin.get("server")
        name = origin.get("tool")
        if isinstance(server, str) and server and isinstance(name, str) and name:
            return server, name
        return None
    return None


def mcp_read_only_hint_for(qualified_name: str, tools: list[dict[str, Any]] | None) -> bool:
    """True if the discovered tool for ``qualified_name`` carries ``readOnlyHint: true``.

    Reads :data:`MCP_ANNOTATIONS_KEY` off the internal tool-dict envelope
    (populated by :func:`make_function_tool` from the MCP server's advertised
    ``Tool.annotations``). Returns ``False`` for an unknown name, a tool with
    no annotations, or ``readOnlyHint`` absent/false — callers never auto-loosen
    on a missing signal.
    """
    for tool in tools or []:
        function = tool.get("function") or {}
        if function.get("name") != qualified_name:
            continue
        annotations = tool.get(MCP_ANNOTATIONS_KEY)
        if not isinstance(annotations, dict):
            return False
        return bool(annotations.get("readOnlyHint"))
    return False
