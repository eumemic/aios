"""Value-domain models for lane lock files and activation results.

These are pure data models — no DB, no service imports. They mirror the
lock-file JSON shape produced by the lane builder (``lane-expand``) and
consumed by the ``lane_activate`` workflow script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Lock-file section models ────────────────────────────────────────────────


@dataclass(frozen=True)
class LockProvenance:
    """``_provenance`` section — builder metadata, not applied to any live object."""
    builder_version: str
    resolved_agent_ids: dict[str, str] = field(default_factory=dict)
    spec_hash: str = ""


@dataclass(frozen=True)
class LockWorkflow:
    """``workflow`` section — the workflow definition to create/update."""
    name: str
    script: str
    description: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    http_servers: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LockCronTriggerAction:
    """``cron_trigger.action`` — the WorkflowAction inside the trigger."""
    workflow_id: str
    input_template: dict[str, Any] | None = None
    vault_ids: list[str] = field(default_factory=list)
    workflow_version: int | None = None


@dataclass(frozen=True)
class LockCronTriggerSource:
    """``cron_trigger.source`` — the CronSource inside the trigger."""
    schedule: str
    timezone: str = "UTC"


@dataclass(frozen=True)
class LockCronTrigger:
    """``cron_trigger`` section — the cron trigger to create/update."""
    name: str
    trigger_name: str
    action: LockCronTriggerAction
    source: LockCronTriggerSource
    enabled: bool = True


@dataclass(frozen=True)
class LockLauncherAgent:
    """``launcher_agent`` section — the agent definition to create/update."""
    name: str
    model: str
    description: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    http_servers: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LockLauncherSession:
    """``launcher_session`` section — the session to create/update."""
    agent_id: str
    environment_id: str
    title: str | None = None
    archive_when_idle: bool = False
    vault_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LaneLock:
    """A fully parsed lane lock file."""
    provenance: LockProvenance
    workflow: LockWorkflow
    cron_trigger: LockCronTrigger
    launcher_agent: LockLauncherAgent
    launcher_session: LockLauncherSession

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaneLock:
        """Parse a lock-file dict into a typed ``LaneLock``."""
        prov = data.get("_provenance", {})
        wf = data["workflow"]
        ct = data["cron_trigger"]
        la = data["launcher_agent"]
        ls = data["launcher_session"]

        return cls(
            provenance=LockProvenance(
                builder_version=prov.get("builder_version", ""),
                resolved_agent_ids=prov.get("resolved_agent_ids", {}),
                spec_hash=prov.get("spec_hash", ""),
            ),
            workflow=LockWorkflow(
                name=wf["name"],
                script=wf["script"],
                description=wf.get("description"),
                tools=wf.get("tools", []),
                http_servers=wf.get("http_servers", []),
            ),
            cron_trigger=LockCronTrigger(
                name=ct["name"],
                trigger_name=ct["trigger_name"],
                action=LockCronTriggerAction(
                    workflow_id=ct["action"]["workflow_id"],
                    input_template=ct["action"].get("input_template"),
                    vault_ids=ct["action"].get("vault_ids", []),
                    workflow_version=ct["action"].get("workflow_version"),
                ),
                source=LockCronTriggerSource(
                    schedule=ct["source"]["schedule"],
                    timezone=ct["source"].get("timezone", "UTC"),
                ),
                enabled=ct.get("enabled", True),
            ),
            launcher_agent=LockLauncherAgent(
                name=la["name"],
                model=la["model"],
                description=la.get("description"),
                tools=la.get("tools", []),
                http_servers=la.get("http_servers", []),
            ),
            launcher_session=LockLauncherSession(
                agent_id=ls["agent_id"],
                environment_id=ls["environment_id"],
                title=ls.get("title"),
                archive_when_idle=ls.get("archive_when_idle", False),
                vault_ids=ls.get("vault_ids", []),
            ),
        )


# ── Activation result models ────────────────────────────────────────────────


class ActivationOutcome(str, Enum):
    """Top-level outcome of a lane activation run."""
    ACTIVATED = "activated"
    NO_OP = "no_op"
    FAILED = "failed"


@dataclass
class ObjectDelta:
    """What changed (or didn't) for one live object."""
    object_kind: str   # "workflow" | "agent" | "session" | "trigger"
    object_name: str
    action: str        # "created" | "updated" | "unchanged"
    object_id: str | None = None
    old_version: int | None = None
    new_version: int | None = None
    error: str | None = None


@dataclass
class ActivationResult:
    """The typed return value of a ``lane_activate`` run."""
    outcome: ActivationOutcome
    lane: str
    merge_sha: str
    spec_hash: str
    deltas: list[ObjectDelta] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "lane": self.lane,
            "merge_sha": self.merge_sha,
            "spec_hash": self.spec_hash,
            "deltas": [
                {
                    "object_kind": d.object_kind,
                    "object_name": d.object_name,
                    "action": d.action,
                    "object_id": d.object_id,
                    "old_version": d.old_version,
                    "new_version": d.new_version,
                    "error": d.error,
                }
                for d in self.deltas
            ],
            "verification": self.verification,
            "error": self.error,
        }
