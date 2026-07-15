from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.wf_run_create_workspace import WfRunCreateWorkspace
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.inline_script_body import InlineScriptBody


T = TypeVar("T", bound="WfRunCreate")


@_attrs_define
class WfRunCreate:
    """Request body for ``POST /v1/runs`` — launch a run.

    Exactly ONE source arm (validated below):

    * ``workflow_id`` (+ optional ``version``) — the registered path: snapshot a
      pre-registered workflow's script + declared surface.
    * ``inline`` (:class:`InlineScriptBody`) — the inline-script arm (T5, #1466): a
      one-shot run launched from an inline ``{script, schemas, surface}`` body with NO
      ``workflows`` row created. ``version`` is meaningless on this arm (no definition
      history) and is rejected if combined with it.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
    ride in request bodies; the HTTP path is always an operator launch.)

        Attributes:
            environment_id (str):
            workflow_id (None | str | Unset): The registered workflow to run. Supply EITHER this or `inline` (exactly one).
                Omit when launching an inline one-shot run.
            inline (InlineScriptBody | None | Unset): Inline-script body for an anonymous one-shot run (T5). Supply EITHER
                this or `workflow_id` (exactly one). No `workflows` row is created.
            workspace (WfRunCreateWorkspace | Unset): Workspace mode. HTTP launches default to a fresh run workspace.
                Default: WfRunCreateWorkspace.FRESH.
            version (int | None | Unset): Optional historical workflow version to run. `None` (default) launches the
                workflow's CURRENT version. An integer re-runs that specific version: the run snapshots that version's script +
                declared surface (clamped against the current launcher's authority) and binds `source_version` to it. Launching
                ANY version of an archived workflow is refused (409). This is a SELECTOR — distinct from the trigger's
                `workflow_version` drift assertion. Not valid with `inline`.
            input_ (Any | Unset):
            vault_ids (list[str] | Unset): Vault ids to bind to the run for credential resolution. When an agent launches
                the run, these must be a subset of the launcher's own vaults; the HTTP path is unattenuated operator authority.
            budget_usd (float | None | Unset): Optional shared USD spend ceiling for this run's direct agent() children.
            default_child_model (None | str | Unset): Optional model used by generic agent() children when they omit model=.
    """

    environment_id: str
    workflow_id: None | str | Unset = UNSET
    inline: InlineScriptBody | None | Unset = UNSET
    workspace: WfRunCreateWorkspace | Unset = WfRunCreateWorkspace.FRESH
    version: int | None | Unset = UNSET
    input_: Any | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    budget_usd: float | None | Unset = UNSET
    default_child_model: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.inline_script_body import InlineScriptBody

        environment_id = self.environment_id

        workflow_id: None | str | Unset
        if isinstance(self.workflow_id, Unset):
            workflow_id = UNSET
        else:
            workflow_id = self.workflow_id

        inline: dict[str, Any] | None | Unset
        if isinstance(self.inline, Unset):
            inline = UNSET
        elif isinstance(self.inline, InlineScriptBody):
            inline = self.inline.to_dict()
        else:
            inline = self.inline

        workspace: str | Unset = UNSET
        if not isinstance(self.workspace, Unset):
            workspace = self.workspace.value

        version: int | None | Unset
        if isinstance(self.version, Unset):
            version = UNSET
        else:
            version = self.version

        input_ = self.input_

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        budget_usd: float | None | Unset
        if isinstance(self.budget_usd, Unset):
            budget_usd = UNSET
        else:
            budget_usd = self.budget_usd

        default_child_model: None | str | Unset
        if isinstance(self.default_child_model, Unset):
            default_child_model = UNSET
        else:
            default_child_model = self.default_child_model

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "environment_id": environment_id,
            }
        )
        if workflow_id is not UNSET:
            field_dict["workflow_id"] = workflow_id
        if inline is not UNSET:
            field_dict["inline"] = inline
        if workspace is not UNSET:
            field_dict["workspace"] = workspace
        if version is not UNSET:
            field_dict["version"] = version
        if input_ is not UNSET:
            field_dict["input"] = input_
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if budget_usd is not UNSET:
            field_dict["budget_usd"] = budget_usd
        if default_child_model is not UNSET:
            field_dict["default_child_model"] = default_child_model

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.inline_script_body import InlineScriptBody

        d = dict(src_dict)
        environment_id = d.pop("environment_id")

        def _parse_workflow_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        workflow_id = _parse_workflow_id(d.pop("workflow_id", UNSET))

        def _parse_inline(data: object) -> InlineScriptBody | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inline_type_0 = InlineScriptBody.from_dict(data)

                return inline_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(InlineScriptBody | None | Unset, data)

        inline = _parse_inline(d.pop("inline", UNSET))

        _workspace = d.pop("workspace", UNSET)
        workspace: WfRunCreateWorkspace | Unset
        if isinstance(_workspace, Unset):
            workspace = UNSET
        else:
            workspace = WfRunCreateWorkspace(_workspace)

        def _parse_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        version = _parse_version(d.pop("version", UNSET))

        input_ = d.pop("input", UNSET)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        def _parse_budget_usd(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        budget_usd = _parse_budget_usd(d.pop("budget_usd", UNSET))

        def _parse_default_child_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        default_child_model = _parse_default_child_model(
            d.pop("default_child_model", UNSET)
        )

        wf_run_create = cls(
            environment_id=environment_id,
            workflow_id=workflow_id,
            inline=inline,
            workspace=workspace,
            version=version,
            input_=input_,
            vault_ids=vault_ids,
            budget_usd=budget_usd,
            default_child_model=default_child_model,
        )

        return wf_run_create
