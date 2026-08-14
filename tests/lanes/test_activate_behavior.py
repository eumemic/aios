"""Behavioural tests for the ``lane_activate`` script — these EXECUTE ``main()``.

The other lane tests are structural (they string-match and compile the script
constant). Nothing there runs the script, so none of them can observe what the
activation actually *does*. These tests execute the real
``LANE_ACTIVATE_SCRIPT`` against an in-memory fake of the aios + GitHub APIs
that enforces the properties the real ones do: name uniqueness, optimistic
concurrency on update, and archived-session filtering.

They pin the three properties that structural tests cannot reach:

1. idempotency — re-run and concurrent-run convergence (``TestIdempotency``)
2. partial apply — a half-activated lane is detectable and self-repairing
   on re-run (``TestPartialApply``)
3. verification is load-bearing — a failed OR unperformable post-apply check
   FAILS the activation rather than being collected and ignored
   (``TestVerificationFailsLoudly``)
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT


def _lock(script: str = "S") -> dict[str, Any]:
    return {
        "_provenance": {"builder_version": "lane-expand/3", "spec_hash": "spec-abc"},
        "workflow": {
            "name": "__lane__wf",
            "script": script,
            "description": "d",
            "tools": [],
            "http_servers": [],
        },
        "cron_trigger": {
            "name": "__lane__trig",
            "trigger_name": "trig",
            "action": {
                "workflow_id": "__lane__wf",
                "input_template": {},
                "vault_ids": [],
            },
            "source": {"schedule": "0 * * * *", "timezone": "UTC"},
            "enabled": True,
        },
        "launcher_agent": {
            "name": "__lane__ag",
            "model": "m",
            "description": "d",
            "tools": [],
            "http_servers": [],
        },
        "launcher_session": {
            "agent_id": "__lane__ag",
            "environment_id": "env",
            "title": "t",
            "archive_when_idle": False,
            "vault_ids": [],
        },
    }


def _ok(obj: Any) -> dict[str, Any]:
    return {"status": 200, "body": json.dumps(obj)}


class FakeWorld:
    """In-memory aios + GitHub with name uniqueness and optimistic concurrency."""

    def __init__(
        self,
        *,
        lock: dict[str, Any] | None = None,
        trigger_create_fails: bool = False,
        telemetry_path_ok: bool = True,
        trigger_enabled: bool = True,
        deployed_script_has_repo: bool = True,
        workflow_get_fails: bool = False,
    ) -> None:
        self.lock = lock if lock is not None else _lock()
        self.trigger_create_fails = trigger_create_fails
        self.telemetry_path_ok = telemetry_path_ok
        self.trigger_enabled = trigger_enabled
        self.deployed_script_has_repo = deployed_script_has_repo
        self.workflow_get_fails = workflow_get_fails
        self.workflows: dict[str, Any] = {}
        self.agents: dict[str, Any] = {}
        self.sessions: dict[str, Any] = {}
        self.triggers: dict[str, Any] = {}

    def tool(self):  # noqa: C901 - a dispatch table, flat by nature
        async def _tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            method, path = args["method"], args["path"]
            payload = json.loads(args["body"]) if args.get("body") else None

            if args["server_ref"] == "github":
                if "/contents/app/infra/lanes/" in path:
                    raw = base64.b64encode(json.dumps(self.lock).encode()).decode()
                    return _ok({"content": raw})
                if "resource_telemetry.json" in path:
                    if not self.telemetry_path_ok:
                        return {"status": 404, "body": json.dumps({"detail": "absent"})}
                    return _ok({"path": "ops/telemetry/resource_telemetry.json"})
                return _ok({})

            if method == "GET" and path.startswith("/v1/workflows?"):
                return _ok({"data": list(self.workflows.values())})
            if method == "POST" and path == "/v1/workflows":
                if payload["name"] in self.workflows:
                    return {"status": 409, "body": json.dumps({"detail": "duplicate"})}
                self.workflows[payload["name"]] = {
                    "id": "wf-1",
                    "name": payload["name"],
                    "version": 1,
                    "script": payload["script"],
                    "description": payload.get("description"),
                    "tools": [],
                    "http_servers": [],
                }
                return _ok(self.workflows[payload["name"]])
            if method == "PUT" and path.startswith("/v1/workflows/"):
                cur = self.workflows[payload["name"]]
                if payload["version"] != cur["version"]:
                    return {"status": 409, "body": json.dumps({"detail": "conflict"})}
                cur.update(script=payload["script"], version=cur["version"] + 1)
                return _ok(cur)
            if method == "GET" and path.startswith("/v1/workflows/"):
                if self.workflow_get_fails:
                    return {"status": 500, "body": json.dumps({"detail": "boom"})}
                live = dict(self.workflows.get("__lane__wf", {}))
                if not live:
                    return {"status": 404, "body": json.dumps({"detail": "gone"})}
                if not self.deployed_script_has_repo:
                    live["script"] = "a script with no telemetry repo reference"
                else:
                    live["script"] = "targets eumemic/eumemic-company"
                return _ok(live)

            if method == "GET" and path.startswith("/v1/agents?"):
                return _ok({"data": list(self.agents.values())})
            if method == "POST" and path == "/v1/agents":
                if payload["name"] in self.agents:
                    return {"status": 409, "body": json.dumps({"detail": "duplicate"})}
                self.agents[payload["name"]] = {
                    "id": "ag-1",
                    "name": payload["name"],
                    "version": 1,
                    "model": payload["model"],
                    "description": payload.get("description"),
                    "tools": [],
                    "http_servers": [],
                }
                return _ok(self.agents[payload["name"]])

            if method == "GET" and path.startswith("/v1/sessions?"):
                return _ok({"data": list(self.sessions.values())})
            if method == "POST" and path == "/v1/sessions":
                self.sessions["s"] = {
                    "id": "sess-1",
                    "status": "active",
                    "title": payload.get("title"),
                    "vault_ids": [],
                }
                return _ok(self.sessions["s"])

            if method == "GET" and path.endswith("/triggers"):
                return _ok({"data": list(self.triggers.values())})
            if method == "POST" and path.endswith("/triggers"):
                if self.trigger_create_fails:
                    return {"status": 500, "body": json.dumps({"detail": "down"})}
                self.triggers["trig"] = {
                    "id": "tr-1",
                    "name": payload["name"],
                    "enabled": self.trigger_enabled,
                    "next_fire": "2026-01-01T00:00:00Z",
                    "source": payload["source"],
                    "action": payload["action"],
                }
                return _ok(self.triggers["trig"])

            return _ok({})

        return _tool

    def activate(self, merge_sha: str = "sha1") -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        namespace["tool"] = self.tool()
        namespace["log"] = lambda *a, **k: None
        namespace["phase"] = lambda *a, **k: None
        return asyncio.run(
            namespace["main"]({"input": {"lane": "test", "merge_sha": merge_sha}})
        )

    async def activate_async(self, merge_sha: str = "sha1") -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        namespace["tool"] = self.tool()
        namespace["log"] = lambda *a, **k: None
        namespace["phase"] = lambda *a, **k: None
        return await namespace["main"]({"input": {"lane": "test", "merge_sha": merge_sha}})


def _actions(result: dict[str, Any]) -> dict[str, str]:
    return {d["object_kind"]: d["action"] for d in result["deltas"]}


# ── Q1: idempotency ──────────────────────────────────────────────────────────


class TestIdempotency:
    def test_rerun_of_identical_lock_is_a_no_op(self) -> None:
        """Second run creates nothing and reports no_op."""
        world = FakeWorld()

        first = world.activate()
        assert first["outcome"] == "activated"
        assert _actions(first) == {
            "workflow": "created",
            "agent": "created",
            "session": "created",
            "trigger": "created",
        }

        second = world.activate()
        assert second["outcome"] == "no_op"
        assert _actions(second) == {
            "workflow": "unchanged",
            "agent": "unchanged",
            "session": "unchanged",
            "trigger": "unchanged",
        }
        assert len(world.workflows) == 1
        assert len(world.agents) == 1
        assert len(world.sessions) == 1
        assert len(world.triggers) == 1

    def test_concurrent_activations_do_not_duplicate_objects(self) -> None:
        """At-least-once delivery: two runs racing converge on one object set."""
        world = FakeWorld()

        async def race() -> tuple[dict[str, Any], dict[str, Any]]:
            return await asyncio.gather(  # type: ignore[return-value]
                world.activate_async(), world.activate_async()
            )

        first, second = asyncio.run(race())

        assert len(world.workflows) == 1
        assert len(world.agents) == 1
        assert len(world.sessions) == 1
        assert len(world.triggers) == 1
        outcomes = sorted([first["outcome"], second["outcome"]])
        assert outcomes == ["activated", "no_op"]
        for result in (first, second):
            assert "error" not in json.dumps(_actions(result))

    def test_rerun_at_moved_merge_sha_updates_under_optimistic_concurrency(self) -> None:
        """A moved merge_sha converges live state onto the new lock and bumps version."""
        world = FakeWorld()
        world.activate("sha1")
        assert world.workflows["__lane__wf"]["version"] == 1

        world.lock = _lock(script="S-VERSION-2")
        moved = world.activate("sha2")

        assert moved["outcome"] == "activated"
        assert _actions(moved)["workflow"] == "updated"
        assert world.workflows["__lane__wf"]["script"] == "S-VERSION-2"
        assert world.workflows["__lane__wf"]["version"] == 2

    def test_activation_is_last_writer_wins_not_sha_ordered(self) -> None:
        """Documents a REAL characteristic: a replayed OLD merge_sha rolls state back.

        The script converges on whatever lock it is handed; it does not compare
        merge_sha against what is already live. A stale redelivery therefore
        reverts the lane. This is pinned so the behaviour cannot change silently.
        """
        world = FakeWorld()
        world.activate("sha1")
        world.lock = _lock(script="S-VERSION-2")
        world.activate("sha2")
        assert world.workflows["__lane__wf"]["script"] == "S-VERSION-2"

        world.lock = _lock(script="S")
        replay = world.activate("sha1")

        assert replay["outcome"] == "activated"
        assert world.workflows["__lane__wf"]["script"] == "S"
        assert world.workflows["__lane__wf"]["version"] == 3


# ── Q2: partial apply ────────────────────────────────────────────────────────


class TestPartialApply:
    def test_trigger_failure_reports_failed_with_the_partial_deltas(self) -> None:
        """A half-activated lane is DETECTABLE: outcome failed + per-object deltas."""
        world = FakeWorld(trigger_create_fails=True)

        result = world.activate()

        assert result["outcome"] == "failed"
        assert "trigger backend" in result["error"] or "down" in result["error"]
        assert _actions(result) == {
            "workflow": "created",
            "agent": "created",
            "session": "created",
            "trigger": "error",
        }
        assert len(world.workflows) == 1
        assert len(world.triggers) == 0

    def test_partial_apply_is_repaired_by_plain_rerun(self) -> None:
        """No manual repair: re-running completes the missing object only."""
        world = FakeWorld(trigger_create_fails=True)
        assert world.activate()["outcome"] == "failed"

        world.trigger_create_fails = False
        repair = world.activate()

        assert repair["outcome"] == "activated"
        assert _actions(repair) == {
            "workflow": "unchanged",
            "agent": "unchanged",
            "session": "unchanged",
            "trigger": "created",
        }
        assert len(world.workflows) == 1
        assert len(world.triggers) == 1


# ── Q3: verification must be load-bearing ────────────────────────────────────


class TestVerificationFailsLoudly:
    def test_healthy_activation_passes_all_checks(self) -> None:
        result = FakeWorld().activate()

        assert result["outcome"] == "activated"
        assert result["error"] is None
        assert result["verification"] == {
            "telemetry_repo_in_script": True,
            "trigger_enabled": True,
            "trigger_next_fire": True,
            "telemetry_path_exists": True,
        }

    def test_missing_telemetry_path_fails_the_activation(self) -> None:
        """A collected False must not be reported as a successful activation."""
        result = FakeWorld(telemetry_path_ok=False).activate()

        assert result["verification"]["telemetry_path_exists"] is False
        assert result["outcome"] == "failed"
        assert "telemetry_path_exists" in result["error"]

    def test_disabled_trigger_fails_the_activation(self) -> None:
        result = FakeWorld(trigger_enabled=False).activate()

        assert result["verification"]["trigger_enabled"] is False
        assert result["outcome"] == "failed"
        assert "trigger_enabled" in result["error"]

    def test_deployed_script_missing_telemetry_repo_fails_the_activation(self) -> None:
        result = FakeWorld(deployed_script_has_repo=False).activate()

        assert result["verification"]["telemetry_repo_in_script"] is False
        assert result["outcome"] == "failed"
        assert "telemetry_repo_in_script" in result["error"]

    def test_unperformable_check_fails_the_activation(self) -> None:
        """None means 'could not verify'. Unverifiable is not healthy."""
        result = FakeWorld(workflow_get_fails=True).activate()

        assert result["verification"]["telemetry_repo_in_script"] is None
        assert result["outcome"] == "failed"
        assert "telemetry_repo_in_script" in result["error"]

    def test_failed_activation_still_reports_full_verification_and_deltas(self) -> None:
        """The failure is diagnosable, not just a bare error string."""
        result = FakeWorld(telemetry_path_ok=False).activate()

        assert result["outcome"] == "failed"
        assert result["verification"]["telemetry_repo_in_script"] is True
        assert result["verification"]["telemetry_path_exists"] is False
        assert _actions(result)["workflow"] == "created"
        assert result["spec_hash"] == "spec-abc"


class TestFailedChecksHelper:
    """Direct unit tests of the failed_checks predicate inside the script."""

    @staticmethod
    def _failed_checks():
        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        return namespace["failed_checks"]

    def test_all_true_is_empty(self) -> None:
        assert self._failed_checks()({"a": True, "b": True}) == []

    def test_false_and_none_both_fail(self) -> None:
        assert self._failed_checks()({"a": True, "b": False, "c": None}) == ["b", "c"]

    def test_diagnostic_error_keys_are_not_treated_as_checks(self) -> None:
        checks = {"trigger_enabled": True, "trigger_error": "some detail"}
        assert self._failed_checks()(checks) == []
