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
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, cast

from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT
from aios.models.common import ListResponse
from aios.models.pagination import DEFAULT_PAGE_LIMIT, decode_cursor


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
        seed_sessions: list[dict[str, Any]] | None = None,
        seed_triggers: list[dict[str, Any]] | None = None,
        session_page_limit: int = DEFAULT_PAGE_LIMIT,
        session_list_fails_after_page: int | None = None,
    ) -> None:
        self.lock = lock if lock is not None else _lock()
        # GET /v1/sessions is keyset-paginated at DEFAULT_PAGE_LIMIT. Modelling it
        # as one unbounded page (what this fake used to do) makes the fake
        # STRUCTURALLY unable to exhibit a first-page-only scan bug.
        self.session_page_limit = session_page_limit
        # Every ?cursor= token the script sent back, so a test can assert the
        # walk actually happened rather than inferring it from the outcome.
        self.session_list_cursors: list[str | None] = []
        # Make page N+1 of the session scan unreadable, so a test can pin that an
        # UNPROVABLE list fails instead of silently reading as "not found".
        self.session_list_fails_after_page = session_list_fails_after_page
        self.trigger_create_fails = trigger_create_fails
        self.telemetry_path_ok = telemetry_path_ok
        self.trigger_enabled = trigger_enabled
        self.deployed_script_has_repo = deployed_script_has_repo
        self.workflow_get_fails = workflow_get_fails
        self.workflows: dict[str, Any] = {}
        self.agents: dict[str, Any] = {}
        self.sessions: dict[str, Any] = {}
        self.triggers: dict[str, Any] = {}
        # Payloads the script actually PUT, so tests can assert on the wire shape.
        self.session_updates: list[dict[str, Any]] = []
        self.trigger_updates: list[dict[str, Any]] = []
        for sess in seed_sessions or []:
            self.sessions[sess["id"]] = dict(sess)
        for trig in seed_triggers or []:
            self.triggers[trig["name"]] = dict(trig)

    def tool(self) -> Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]:
        async def _tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            method, path = args["method"], args["path"]
            payload: dict[str, Any] = json.loads(args["body"]) if args.get("body") else {}

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
                return self._list_sessions(path)
            if method == "POST" and path == "/v1/sessions":
                self.sessions["s"] = {
                    "id": "sess-1",
                    "status": "active",
                    "agent_id": payload.get("agent_id"),
                    "environment_id": payload.get("environment_id"),
                    "archive_when_idle": payload.get("archive_when_idle", False),
                    "title": payload.get("title"),
                    "vault_ids": payload.get("vault_ids", []),
                }
                return _ok(self.sessions["s"])
            if method == "PUT" and path.startswith("/v1/sessions/") and "/triggers/" not in path:
                self.session_updates.append(payload)
                sid = path.rsplit("/", 1)[1]
                for sess in self.sessions.values():
                    if sess["id"] == sid:
                        sess.update(
                            {k: v for k, v in payload.items() if k != "version"},
                        )
                        return _ok(sess)
                return {"status": 404, "body": json.dumps({"detail": "no session"})}

            if method == "GET" and path.endswith("/triggers"):
                return _ok({"data": list(self.triggers.values())})
            if method == "PUT" and "/triggers/" in path:
                self.trigger_updates.append(payload)
                name = path.rsplit("/", 1)[1]
                if name not in self.triggers:
                    return {"status": 404, "body": json.dumps({"detail": "no trigger"})}
                live_trig = self.triggers[name]
                live_trig.update(
                    source=payload["source"],
                    action=payload["action"],
                    enabled=payload.get("enabled", True),
                )
                return _ok(live_trig)
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

    def _list_sessions(self, path: str) -> dict[str, Any]:
        """Keyset-paginated GET /v1/sessions, matching the real router.

        Mirrors ``aios.api.routers.sessions.list_`` + ``ListResponse.paginate``:
        a first page carries ``?agent_id=``/``?limit=``; every later page is
        ``?cursor=<next_cursor>`` ALONE (the token carries the filters back), and
        a page past the end has ``has_more: false`` / ``next_cursor: null``.
        Envelope + token are produced by the REAL production codec, not a
        hand-rolled imitation, so the fake cannot drift from the contract.
        """
        if (
            self.session_list_fails_after_page is not None
            and len(self.session_list_cursors) >= self.session_list_fails_after_page
        ):
            self.session_list_cursors.append("<failed>")
            return {"status": 500, "body": json.dumps({"detail": "sessions backend down"})}

        query = path.split("?", 1)[1]
        params = dict(
            cast(tuple[str, str], tuple(kv.split("=", 1))) for kv in query.split("&") if "=" in kv
        )

        if "cursor" in params:
            # The real router 422s if a cursor is mixed with other params.
            if sorted(params) != ["cursor"]:
                return {
                    "status": 422,
                    "body": json.dumps(
                        {"detail": "A '?cursor=' request takes no other pagination params."}
                    ),
                }
            self.session_list_cursors.append(params["cursor"])
            state = decode_cursor(params["cursor"])
            want = state.filters.get("agent_id")
            after: str | None = str(state.cursor)
            limit = state.limit
        else:
            self.session_list_cursors.append(None)
            want = params.get("agent_id")
            after = None
            limit = int(params.get("limit", self.session_page_limit))

        # Newest-first (DESC by id), which is what the real endpoint returns.
        rows = sorted(
            (s for s in self.sessions.values() if s.get("agent_id") == want),
            key=lambda s: cast(str, s["id"]),
            reverse=True,
        )
        if after is not None:
            rows = [s for s in rows if cast(str, s["id"]) < after]

        page = ListResponse[dict[str, Any]].paginate(
            rows[: limit + 1],
            limit,
            cursor=lambda s: cast(str, s["id"]),
            filters={"agent_id": want},
        )
        return _ok(page.model_dump())

    def activate(self, merge_sha: str = "sha1") -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        namespace["tool"] = self.tool()
        namespace["log"] = lambda *a, **k: None
        namespace["phase"] = lambda *a, **k: None
        main = cast(
            Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]], namespace["main"]
        )
        return asyncio.run(main({"input": {"lane": "test", "merge_sha": merge_sha}}))

    async def activate_async(self, merge_sha: str = "sha1") -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        namespace["tool"] = self.tool()
        namespace["log"] = lambda *a, **k: None
        namespace["phase"] = lambda *a, **k: None
        main = cast(Callable[[dict[str, Any]], Awaitable[dict[str, Any]]], namespace["main"])
        return await main({"input": {"lane": "test", "merge_sha": merge_sha}})


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
            return await asyncio.gather(world.activate_async(), world.activate_async())

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
    def _failed_checks() -> Callable[[dict[str, Any]], list[str]]:
        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        return cast(Callable[[dict[str, Any]], list[str]], namespace["failed_checks"])

    def test_all_true_is_empty(self) -> None:
        assert self._failed_checks()({"a": True, "b": True}) == []

    def test_false_and_none_both_fail(self) -> None:
        assert self._failed_checks()({"a": True, "b": False, "c": None}) == ["b", "c"]

    def test_diagnostic_error_keys_are_not_treated_as_checks(self) -> None:
        checks = {"trigger_enabled": True, "trigger_error": "some detail"}
        assert self._failed_checks()(checks) == []


# ── Remaining review findings (aios#2063, verdict 2026-08-14) ────────────────


def _sess(
    sid: str,
    *,
    title: str | None,
    agent: str = "ag-1",
    vault_ids: list[str] | None = None,
    environment_id: str = "env",
    archive_when_idle: bool = False,
    status: str = "active",
) -> dict[str, Any]:
    return {
        "id": sid,
        "status": status,
        "agent_id": agent,
        "environment_id": environment_id,
        "archive_when_idle": archive_when_idle,
        "title": title,
        "vault_ids": vault_ids if vault_ids is not None else [],
    }


class TestSessionIdentity:
    """Finding 1: activation must not adopt/mutate an unrelated session.

    An agent may legitimately own many active sessions. Selecting 'the first
    non-archived one' can overwrite a bystander session's title and vault
    bindings and attach the lane's cron trigger to it.
    """

    def test_unrelated_session_is_not_hijacked(self) -> None:
        """A pre-existing unrelated session must be left completely untouched."""
        bystander = _sess("sess-other", title="someone else's work", vault_ids=["v-private"])
        world = FakeWorld(seed_sessions=[bystander])

        result = world.activate()

        untouched = world.sessions["sess-other"]
        assert untouched["title"] == "someone else's work"
        assert untouched["vault_ids"] == ["v-private"]
        assert world.session_updates == []
        # The lane got its own session rather than adopting the bystander.
        assert _actions(result)["session"] == "created"
        lane_ids = {s["id"] for s in world.sessions.values() if s["title"] == "t"}
        assert lane_ids and "sess-other" not in lane_ids

    def test_lane_session_is_readopted_not_duplicated(self) -> None:
        """The lane's OWN session (matching lane identity) is still adopted."""
        lane = _sess("sess-lane", title="t")
        world = FakeWorld(seed_sessions=[_sess("sess-other", title="unrelated"), lane])

        result = world.activate()

        assert _actions(result)["session"] == "unchanged"
        assert len(world.sessions) == 2  # nothing new created

    def test_immutable_environment_drift_fails_loudly(self) -> None:
        """environment_id cannot be changed by update; silent acceptance is wrong."""
        drifted = _sess("sess-lane", title="t", environment_id="env-DIFFERENT")
        world = FakeWorld(seed_sessions=[drifted])

        result = world.activate()

        assert result["outcome"] == "failed"
        assert "environment_id" in result["error"]

    def test_archive_when_idle_drift_is_reconciled(self) -> None:
        """archive_when_idle drift must be detected, not silently accepted."""
        drifted = _sess("sess-lane", title="t", archive_when_idle=True)
        world = FakeWorld(seed_sessions=[drifted])

        result = world.activate()

        assert _actions(result)["session"] == "updated"
        assert world.sessions["sess-lane"]["archive_when_idle"] is False


def _live_trigger(
    *,
    schedule: str = "0 * * * *",
    timezone: str = "UTC",
    workflow_version: int | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "id": "tr-1",
        "name": "trig",
        "enabled": enabled,
        "next_fire": "2026-01-01T00:00:00Z",
        "source": {"kind": "cron", "schedule": schedule, "timezone": timezone},
        "action": {
            "kind": "workflow",
            "workflow_id": "wf-1",
            "input_template": {},
            "vault_ids": [],
            "workflow_version": workflow_version,
        },
    }


class TestTriggerDriftDetection:
    """Finding 2: drift comparison omits timezone and workflow_version."""

    def test_timezone_drift_is_detected(self) -> None:
        lock = _lock()
        lock["cron_trigger"]["source"]["timezone"] = "America/Los_Angeles"
        world = FakeWorld(lock=lock, seed_triggers=[_live_trigger(timezone="UTC")])

        result = world.activate()

        assert _actions(result)["trigger"] == "updated"
        assert world.triggers["trig"]["source"]["timezone"] == "America/Los_Angeles"

    def test_workflow_version_drift_is_detected(self) -> None:
        lock = _lock()
        lock["cron_trigger"]["action"]["workflow_version"] = 7
        world = FakeWorld(lock=lock, seed_triggers=[_live_trigger(workflow_version=3)])

        result = world.activate()

        assert _actions(result)["trigger"] == "updated"
        assert world.triggers["trig"]["action"]["workflow_version"] == 7

    def test_matching_trigger_is_still_unchanged(self) -> None:
        """The drift fix must not make every run report a spurious update."""
        lock = _lock()
        lock["cron_trigger"]["source"]["timezone"] = "America/Los_Angeles"
        lock["cron_trigger"]["action"]["workflow_version"] = 7
        world = FakeWorld(
            lock=lock,
            seed_triggers=[_live_trigger(timezone="America/Los_Angeles", workflow_version=7)],
        )

        result = world.activate()

        assert _actions(result)["trigger"] == "unchanged"
        assert world.trigger_updates == []

    def test_replace_payload_carries_no_undefined_version_field(self) -> None:
        """The action never models `version`; emitting version=None is a bug."""
        lock = _lock()
        lock["cron_trigger"]["action"]["workflow_version"] = 7
        world = FakeWorld(lock=lock, seed_triggers=[_live_trigger(workflow_version=3)])

        world.activate()

        assert world.trigger_updates, "expected a trigger PUT"
        assert "version" not in world.trigger_updates[0]["action"]


# ── Fix round 2 (aios#2063): consumer-side guards on lane-session lookup ─────


class TestNullTitleIsRejected:
    """Defect 1: a null/empty lane title must never be used as an identity key.

    ``LockLauncherSession.title`` is ``str | None = None``, so a lock can carry
    no title at all. Scanning with that key makes ``item.get("title") == title``
    true for ANY untitled session — ``None == None`` — so activation adopts an
    unrelated bystander, overwrites its vault bindings, and hangs the lane's
    cron trigger off it. The producer (lane-expand) lives in another repo, so
    this is guarded HERE, at the consumer, regardless of what it promises.
    """

    def test_null_title_lock_does_not_adopt_an_untitled_bystander(self) -> None:
        lock = _lock()
        lock["launcher_session"]["title"] = None
        bystander = _sess("sess-untitled", title=None, vault_ids=["v-private"])
        world = FakeWorld(lock=lock, seed_sessions=[bystander])

        result = world.activate()

        # Hard, loud failure — not a silent adoption.
        assert result["outcome"] == "failed"
        assert _actions(result)["session"] == "error"
        assert "title" in result["error"]
        # The bystander is untouched and no lane session was invented.
        assert world.sessions["sess-untitled"]["vault_ids"] == ["v-private"]
        assert world.session_updates == []
        assert len(world.sessions) == 1

    def test_missing_title_key_is_rejected(self) -> None:
        """An absent ``title`` key is the same defect as an explicit null."""
        lock = _lock()
        del lock["launcher_session"]["title"]
        world = FakeWorld(lock=lock, seed_sessions=[_sess("sess-untitled", title=None)])

        result = world.activate()

        assert result["outcome"] == "failed"
        assert _actions(result)["session"] == "error"
        assert world.session_updates == []

    def test_blank_title_is_rejected(self) -> None:
        """A whitespace-only title is not a usable identity key either."""
        lock = _lock()
        lock["launcher_session"]["title"] = "   "
        world = FakeWorld(lock=lock, seed_sessions=[_sess("sess-blank", title="   ")])

        result = world.activate()

        assert result["outcome"] == "failed"
        assert _actions(result)["session"] == "error"
        assert world.session_updates == []

    def test_guard_does_not_fire_on_a_healthy_titled_lock(self) -> None:
        """NEGATIVE CONTROL: a valid titled lock still activates cleanly."""
        world = FakeWorld(seed_sessions=[_sess("sess-other", title=None)])

        result = world.activate()

        assert result["outcome"] == "activated"
        assert result["error"] is None
        assert _actions(result)["session"] == "created"
        # The untitled bystander was neither adopted nor modified.
        assert world.session_updates == []
        assert world.sessions["sess-other"]["title"] is None


class TestSessionScanFollowsPagination:
    """Defect 2: a first-page-only scan silently creates a DUPLICATE lane session.

    ``GET /v1/sessions`` is keyset-paginated at ``DEFAULT_PAGE_LIMIT`` and hands
    back an opaque ``next_cursor``. An agent may legitimately own hundreds of
    sessions; if the lane's own session sorts past page one, a first-page-only
    scan reads NOT FOUND and creates a second one — two sessions, two cron
    triggers, one lane.
    """

    @staticmethod
    def _crowd(n: int, *, lane_title: str = "t") -> list[dict[str, Any]]:
        """``n`` bystanders sorting AFTER the lane session on a DESC-by-id scan.

        Ids are zero-padded so ordering is deterministic; the lane session gets
        the lowest id, so it lands on the LAST page.
        """
        crowd = [_sess(f"sess-z{i:04d}", title=f"bystander-{i}") for i in range(n)]
        return [*crowd, _sess("sess-a-lane", title=lane_title)]

    def test_lane_session_beyond_the_first_page_is_found_not_duplicated(self) -> None:
        world = FakeWorld(seed_sessions=self._crowd(120))
        assert len(world.sessions) == 121

        result = world.activate()

        # It must ADOPT the existing lane session, not create a second one.
        assert _actions(result)["session"] == "unchanged"
        assert len(world.sessions) == 121, "a duplicate lane session was created"
        lane_sessions = [s for s in world.sessions.values() if s["title"] == "t"]
        assert len(lane_sessions) == 1
        assert lane_sessions[0]["id"] == "sess-a-lane"

    def test_the_scan_actually_walks_every_page(self) -> None:
        """The walk is real: page 2+ are fetched via the opaque cursor."""
        world = FakeWorld(seed_sessions=self._crowd(120))

        world.activate()

        # 121 rows at 50/page = 3 pages: first page + 2 cursor follow-ups.
        first_scan = world.session_list_cursors[:3]
        assert first_scan[0] is None, "first page must not carry a cursor"
        assert all(c is not None for c in first_scan[1:]), "later pages must use ?cursor="

    def test_exhaustion_stops_at_the_last_page(self) -> None:
        """A single-page result must not spin on a stale cursor."""
        world = FakeWorld(seed_sessions=[_sess("sess-a-lane", title="t")])

        result = world.activate()

        assert _actions(result)["session"] == "unchanged"
        assert world.session_list_cursors == [None]

    def test_a_truly_absent_lane_session_is_still_created(self) -> None:
        """NEGATIVE CONTROL: exhausting the pages without a match still creates."""
        crowd = [_sess(f"sess-z{i:04d}", title=f"bystander-{i}") for i in range(120)]
        world = FakeWorld(seed_sessions=crowd)

        result = world.activate()

        assert result["outcome"] == "activated"
        assert _actions(result)["session"] == "created"
        assert len([s for s in world.sessions.values() if s["title"] == "t"]) == 1

    def test_an_unreadable_later_page_fails_instead_of_creating_a_duplicate(self) -> None:
        """If the list cannot be PROVEN complete, fail — never fall through to create.

        A mid-walk error means the lane session may exist on a page that was
        never read. Reporting not-found there is exactly how a duplicate gets
        created, so an unreadable page is a hard failure.
        """
        world = FakeWorld(seed_sessions=self._crowd(120), session_list_fails_after_page=1)

        result = world.activate()

        assert result["outcome"] == "failed"
        assert _actions(result)["session"] == "error"
        assert "list sessions failed" in result["error"]
        # Crucially: no session was invented while the list was unproven.
        assert len(world.sessions) == 121
