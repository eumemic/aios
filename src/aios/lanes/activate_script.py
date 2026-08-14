"""The ``lane_activate`` workflow script source.

This module holds the script text as a constant; it is NOT executed directly.
The script is registered as a workflow's ``script`` field and runs inside
the aios workflow runtime (``wf_script_host``), where it has access to
``tool()``, ``log()``, ``phase()``, and the other author-namespace capabilities.

The script:
1. Reads the lock file from GitHub at ``merge_sha``
2. For each object (workflow, agent, session, trigger):
   - Finds the live object by name
   - Creates it if missing, updates it if changed (optimistic concurrency)
   - Never deletes
3. Verifies post-apply state
4. Returns a typed result dict
"""

LANE_ACTIVATE_SCRIPT = r'''
# ── lane_activate workflow script ──────────────────────────────────────────
# Runs inside aios wf_script_host. Capabilities: tool(), log(), phase().
# Input: {"trigger": {...}, "input": {"lane": str, "merge_sha": str}}
# Output: ActivationResult dict

import json
import base64

GITHUB_SERVER = "github"
AIOS_SERVER = "aios"
LOCK_PATH_TEMPLATE = "app/infra/lanes/{lane}.lock.json"
TELEMETRY_REPO = "eumemic/eumemic-company"
TELEMETRY_PATH = "ops/telemetry/resource_telemetry.json"


# ── helpers ────────────────────────────────────────────────────────────────

async def gh(method, path, body=None):
    """GitHub API call via the run's bound http_request."""
    args = {"server_ref": GITHUB_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    resp = await tool("http_request", args)
    return resp


async def aios_api(method, path, body=None):
    """aios API call via the run's bound http_request."""
    args = {"server_ref": AIOS_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    resp = await tool("http_request", args)
    return resp


def status(resp):
    """Extract status code from http_request response."""
    if isinstance(resp, dict):
        return resp.get("status", 0)
    return 0


def body(resp):
    """Parse response body as JSON."""
    if isinstance(resp, dict):
        raw = resp.get("body", "")
        if isinstance(raw, str) and raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return raw
    return None


def is_error(resp):
    """True if the response is an error or non-2xx."""
    if isinstance(resp, dict) and "error" in resp:
        return True
    s = status(resp)
    return s < 200 or s >= 300


# ── lock file reader ──────────────────────────────────────────────────────

async def read_lock_file(lane, merge_sha):
    """Read and parse the lock file from GitHub at the given commit."""
    path = LOCK_PATH_TEMPLATE.format(lane=lane)
    resp = await gh("GET", f"/repos/{TELEMETRY_REPO}/contents/{path}?ref={merge_sha}")
    if is_error(resp):
        return None, f"failed to read lock file: {resp}"
    data = body(resp)
    if data is None:
        return None, "lock file response body is empty"
    # GitHub contents API returns base64-encoded content
    content_b64 = data.get("content", "")
    if not content_b64:
        return None, "lock file has no content field"
    try:
        content = base64.b64decode(content_b64.replace("\n", "")).decode("utf-8")
        lock = json.loads(content)
    except Exception as exc:
        return None, f"failed to decode lock file: {exc}"
    return lock, None


# ── object finders ─────────────────────────────────────────────────────────

async def find_workflow_by_name(name):
    """Find a workflow by name. Returns (workflow_dict, None) or (None, error)."""
    resp = await aios_api("GET", f"/v1/workflows?name={name}")
    if is_error(resp):
        return None, f"list workflows failed: {resp}"
    data = body(resp)
    items = data.get("data", []) if data else []
    for item in items:
        if item.get("name") == name:
            return item, None
    return None, None  # not found, not an error


async def find_agent_by_name(name):
    """Find an agent by name. Returns (agent_dict, None) or (None, error)."""
    resp = await aios_api("GET", f"/v1/agents?name={name}")
    if is_error(resp):
        return None, f"list agents failed: {resp}"
    data = body(resp)
    items = data.get("data", []) if data else []
    for item in items:
        if item.get("name") == name:
            return item, None
    return None, None


async def find_session_by_agent_name(agent_name, agent_id):
    """Find a session by its agent_id. Returns (session_dict, None) or (None, error)."""
    resp = await aios_api("GET", f"/v1/sessions?agent_id={agent_id}")
    if is_error(resp):
        return None, f"list sessions failed: {resp}"
    data = body(resp)
    items = data.get("data", []) if data else []
    # Return the first non-archived session for this agent
    for item in items:
        if item.get("status") != "archived":
            return item, None
    return None, None


async def find_trigger_on_session(session_id, trigger_name):
    """Find a trigger by name on a session. Returns (trigger_dict, None) or (None, error)."""
    resp = await aios_api("GET", f"/v1/sessions/{session_id}/triggers")
    if is_error(resp):
        return None, f"list triggers failed: {resp}"
    data = body(resp)
    items = data.get("data", []) if data else []
    for item in items:
        if item.get("name") == trigger_name:
            return item, None
    return None, None


# ── object creators/updaters ──────────────────────────────────────────────

async def ensure_workflow(lock_wf):
    """Create or update the workflow to match the lock. Returns ObjectDelta dict."""
    name = lock_wf["name"]
    log(f"ensure_workflow: {name}")

    live, err = await find_workflow_by_name(name)
    if err:
        return {"object_kind": "workflow", "object_name": name, "action": "error", "error": err}

    if live is None:
        # Create
        create_body = {
            "name": name,
            "script": lock_wf["script"],
            "description": lock_wf.get("description"),
            "tools": lock_wf.get("tools", []),
            "http_servers": lock_wf.get("http_servers", []),
        }
        resp = await aios_api("POST", "/v1/workflows", create_body)
        if is_error(resp):
            return {"object_kind": "workflow", "object_name": name, "action": "error",
                    "error": f"create failed: status={status(resp)} body={body(resp)}"}
        created = body(resp)
        return {"object_kind": "workflow", "object_name": name, "action": "created",
                "object_id": created.get("id"), "new_version": created.get("version")}

    # Update if changed
    live_id = live["id"]
    live_version = live["version"]

    # Check if anything actually changed
    changed = False
    if lock_wf["script"] != live.get("script", ""):
        changed = True
    if lock_wf.get("description") != live.get("description"):
        changed = True
    if lock_wf.get("tools", []) != live.get("tools", []):
        changed = True
    if lock_wf.get("http_servers", []) != live.get("http_servers", []):
        changed = True

    if not changed:
        return {"object_kind": "workflow", "object_name": name, "action": "unchanged",
                "object_id": live_id, "old_version": live_version}

    update_body = {
        "version": live_version,
        "name": name,
        "script": lock_wf["script"],
        "description": lock_wf.get("description"),
        "tools": lock_wf.get("tools", []),
        "http_servers": lock_wf.get("http_servers", []),
    }
    resp = await aios_api("PUT", f"/v1/workflows/{live_id}", update_body)
    if is_error(resp):
        return {"object_kind": "workflow", "object_name": name, "action": "error",
                "error": f"update failed: status={status(resp)} body={body(resp)}"}
    updated = body(resp)
    return {"object_kind": "workflow", "object_name": name, "action": "updated",
            "object_id": live_id, "old_version": live_version,
            "new_version": updated.get("version")}


async def ensure_agent(lock_agent):
    """Create or update the launcher agent to match the lock. Returns ObjectDelta dict."""
    name = lock_agent["name"]
    log(f"ensure_agent: {name}")

    live, err = await find_agent_by_name(name)
    if err:
        return {"object_kind": "agent", "object_name": name, "action": "error", "error": err}

    if live is None:
        create_body = {
            "name": name,
            "model": lock_agent["model"],
            "description": lock_agent.get("description"),
            "tools": lock_agent.get("tools", []),
            "http_servers": lock_agent.get("http_servers", []),
        }
        resp = await aios_api("POST", "/v1/agents", create_body)
        if is_error(resp):
            return {"object_kind": "agent", "object_name": name, "action": "error",
                    "error": f"create failed: status={status(resp)} body={body(resp)}"}
        created = body(resp)
        return {"object_kind": "agent", "object_name": name, "action": "created",
                "object_id": created.get("id"), "new_version": created.get("version")}

    live_id = live["id"]
    live_version = live["version"]

    changed = False
    if lock_agent["model"] != live.get("model", ""):
        changed = True
    if lock_agent.get("description") != live.get("description"):
        changed = True
    if lock_agent.get("tools", []) != live.get("tools", []):
        changed = True
    if lock_agent.get("http_servers", []) != live.get("http_servers", []):
        changed = True

    if not changed:
        return {"object_kind": "agent", "object_name": name, "action": "unchanged",
                "object_id": live_id, "old_version": live_version}

    update_body = {
        "version": live_version,
        "name": name,
        "model": lock_agent["model"],
        "description": lock_agent.get("description"),
        "tools": lock_agent.get("tools", []),
        "http_servers": lock_agent.get("http_servers", []),
    }
    resp = await aios_api("PUT", f"/v1/agents/{live_id}", update_body)
    if is_error(resp):
        return {"object_kind": "agent", "object_name": name, "action": "error",
                "error": f"update failed: status={status(resp)} body={body(resp)}"}
    updated = body(resp)
    return {"object_kind": "agent", "object_name": name, "action": "updated",
            "object_id": live_id, "old_version": live_version,
            "new_version": updated.get("version")}


async def ensure_session(lock_session, agent_id):
    """Create or update the launcher session. Returns (session_id, ObjectDelta dict)."""
    agent_name = lock_session["agent_id"]  # This is the agent NAME in the lock
    log(f"ensure_session for agent: {agent_name} (resolved id: {agent_id})")

    live, err = await find_session_by_agent_name(agent_name, agent_id)
    if err:
        return None, {"object_kind": "session", "object_name": agent_name, "action": "error",
                       "error": err}

    if live is None:
        create_body = {
            "agent_id": agent_id,
            "environment_id": lock_session["environment_id"],
            "title": lock_session.get("title"),
            "archive_when_idle": lock_session.get("archive_when_idle", False),
            "vault_ids": lock_session.get("vault_ids", []),
        }
        resp = await aios_api("POST", "/v1/sessions", create_body)
        if is_error(resp):
            return None, {"object_kind": "session", "object_name": agent_name, "action": "error",
                           "error": f"create failed: status={status(resp)} body={body(resp)}"}
        created = body(resp)
        sid = created.get("id")
        return sid, {"object_kind": "session", "object_name": agent_name, "action": "created",
                      "object_id": sid}

    sid = live["id"]
    # Sessions don't have optimistic concurrency — update vault_ids if needed
    changed = False
    if sorted(lock_session.get("vault_ids", [])) != sorted(live.get("vault_ids", [])):
        changed = True
    if lock_session.get("title") != live.get("title"):
        changed = True

    if not changed:
        return sid, {"object_kind": "session", "object_name": agent_name, "action": "unchanged",
                      "object_id": sid}

    update_body = {
        "title": lock_session.get("title"),
        "vault_ids": lock_session.get("vault_ids", []),
    }
    resp = await aios_api("PUT", f"/v1/sessions/{sid}", update_body)
    if is_error(resp):
        return sid, {"object_kind": "session", "object_name": agent_name, "action": "error",
                      "error": f"update failed: status={status(resp)} body={body(resp)}"}
    return sid, {"object_kind": "session", "object_name": agent_name, "action": "updated",
                  "object_id": sid}


async def ensure_trigger(lock_trigger, session_id, workflow_id):
    """Create or update the cron trigger on the session. Returns ObjectDelta dict."""
    trigger_name = lock_trigger["trigger_name"]
    log(f"ensure_trigger: {trigger_name} on session {session_id}")

    live, err = await find_trigger_on_session(session_id, trigger_name)
    if err:
        return {"object_kind": "trigger", "object_name": trigger_name, "action": "error",
                "error": err}

    # Build the trigger source and action
    source = {"kind": "cron", "schedule": lock_trigger["source"]["schedule"]}
    if lock_trigger["source"].get("timezone", "UTC") != "UTC":
        source["timezone"] = lock_trigger["source"]["timezone"]

    action = {
        "kind": "workflow",
        "workflow_id": workflow_id,
        "input_template": lock_trigger["action"].get("input_template"),
        "vault_ids": lock_trigger["action"].get("vault_ids", []),
        "workflow_version": lock_trigger["action"].get("workflow_version"),
    }

    if live is None:
        create_body = {
            "name": trigger_name,
            "source": source,
            "action": action,
            "enabled": lock_trigger.get("enabled", True),
        }
        resp = await aios_api("POST", f"/v1/sessions/{session_id}/triggers", create_body)
        if is_error(resp):
            return {"object_kind": "trigger", "object_name": trigger_name, "action": "error",
                    "error": f"create failed: status={status(resp)} body={body(resp)}"}
        created = body(resp)
        return {"object_kind": "trigger", "object_name": trigger_name, "action": "created",
                "object_id": created.get("id")}

    # Check if changed
    live_source = live.get("source", {})
    live_action = live.get("action", {})
    changed = False
    if source.get("schedule") != live_source.get("schedule"):
        changed = True
    if action.get("workflow_id") != live_action.get("workflow_id"):
        changed = True
    if action.get("input_template") != live_action.get("input_template"):
        changed = True
    if sorted(action.get("vault_ids", [])) != sorted(live_action.get("vault_ids", [])):
        changed = True
    if lock_trigger.get("enabled", True) != live.get("enabled"):
        changed = True

    if not changed:
        return {"object_kind": "trigger", "object_name": trigger_name, "action": "unchanged",
                "object_id": live.get("id")}

    # TriggerUpdate uses replace semantics for source/action
    update_body = {
        "source": source,
        "action": {
            **action,
            # WorkflowActionReplace requires all fields explicitly
            "workflow_version": action.get("workflow_version"),
            "version": action.get("version"),
            "input_template": action.get("input_template"),
            "vault_ids": action.get("vault_ids", []),
        },
        "enabled": lock_trigger.get("enabled", True),
    }
    resp = await aios_api("PUT", f"/v1/sessions/{session_id}/triggers/{trigger_name}", update_body)
    if is_error(resp):
        return {"object_kind": "trigger", "object_name": trigger_name, "action": "error",
                "error": f"update failed: status={status(resp)} body={body(resp)}"}
    return {"object_kind": "trigger", "object_name": trigger_name, "action": "updated",
            "object_id": live.get("id")}


# ── verification ──────────────────────────────────────────────────────────

async def verify_post_apply(lock, workflow_id, session_id):
    """Post-apply verification checks."""
    checks = {}

    # 1. Verify workflow exists and script contains TELEMETRY_REPO
    wf_resp = await aios_api("GET", f"/v1/workflows/{workflow_id}")
    if not is_error(wf_resp):
        wf_data = body(wf_resp)
        script = wf_data.get("script", "")
        checks["telemetry_repo_in_script"] = TELEMETRY_REPO in script
    else:
        checks["telemetry_repo_in_script"] = None
        checks["telemetry_repo_error"] = str(body(wf_resp))

    # 2. Verify trigger is enabled with next_fire set
    if session_id:
        trigger_name = lock["cron_trigger"]["trigger_name"]
        trigger, err = await find_trigger_on_session(session_id, trigger_name)
        if trigger:
            checks["trigger_enabled"] = trigger.get("enabled", False)
            checks["trigger_next_fire"] = trigger.get("next_fire") is not None
        else:
            checks["trigger_enabled"] = None
            checks["trigger_error"] = err or "trigger not found"

    # 3. Check telemetry path exists in TELEMETRY_REPO
    tel_resp = await gh("GET", f"/repos/{TELEMETRY_REPO}/contents/{TELEMETRY_PATH}")
    checks["telemetry_path_exists"] = not is_error(tel_resp)

    return checks


def failed_checks(checks):
    """Names of verification checks that did not pass.

    A check passes ONLY if it is exactly True. False means the check ran and
    the post-apply state is wrong; None means the check could not be performed
    (the API call errored) — unverifiable is NOT healthy, so both fail.
    Keys ending in ``_error`` carry diagnostic strings, not check results.
    """
    return sorted(
        name
        for name, value in checks.items()
        if not name.endswith("_error") and value is not True
    )


# ── main ──────────────────────────────────────────────────────────────────

async def main(input):
    """Entry point for the lane_activate workflow."""
    # Extract input
    trigger_ctx = input.get("trigger", {})
    user_input = input.get("input", input)  # fallback for direct invocation
    lane = user_input.get("lane")
    merge_sha = user_input.get("merge_sha")

    if not lane or not merge_sha:
        return {
            "outcome": "failed",
            "lane": lane or "",
            "merge_sha": merge_sha or "",
            "spec_hash": "",
            "deltas": [],
            "verification": {},
            "error": "missing required input: lane and merge_sha",
        }

    log(f"lane_activate: lane={lane} merge_sha={merge_sha[:12]}")

    # Phase 1: Read lock file
    phase("read-lock")
    lock, err = await read_lock_file(lane, merge_sha)
    if err:
        return {
            "outcome": "failed",
            "lane": lane,
            "merge_sha": merge_sha,
            "spec_hash": "",
            "deltas": [],
            "verification": {},
            "error": err,
        }

    spec_hash = lock.get("_provenance", {}).get("spec_hash", "")
    log(f"lock file loaded: spec_hash={spec_hash}")

    deltas = []

    # Phase 2: Ensure workflow
    phase("ensure-workflow")
    wf_delta = await ensure_workflow(lock["workflow"])
    deltas.append(wf_delta)
    if wf_delta.get("action") == "error":
        return {
            "outcome": "failed", "lane": lane, "merge_sha": merge_sha,
            "spec_hash": spec_hash, "deltas": deltas, "verification": {},
            "error": wf_delta.get("error"),
        }
    workflow_id = wf_delta.get("object_id")

    # Phase 3: Ensure launcher agent
    phase("ensure-agent")
    agent_delta = await ensure_agent(lock["launcher_agent"])
    deltas.append(agent_delta)
    if agent_delta.get("action") == "error":
        return {
            "outcome": "failed", "lane": lane, "merge_sha": merge_sha,
            "spec_hash": spec_hash, "deltas": deltas, "verification": {},
            "error": agent_delta.get("error"),
        }
    agent_id = agent_delta.get("object_id")

    # Phase 4: Ensure launcher session
    phase("ensure-session")
    session_id, session_delta = await ensure_session(lock["launcher_session"], agent_id)
    deltas.append(session_delta)
    if session_delta.get("action") == "error":
        return {
            "outcome": "failed", "lane": lane, "merge_sha": merge_sha,
            "spec_hash": spec_hash, "deltas": deltas, "verification": {},
            "error": session_delta.get("error"),
        }

    # Phase 5: Ensure cron trigger (needs workflow_id for the action)
    phase("ensure-trigger")
    trigger_delta = await ensure_trigger(lock["cron_trigger"], session_id, workflow_id)
    deltas.append(trigger_delta)
    if trigger_delta.get("action") == "error":
        return {
            "outcome": "failed", "lane": lane, "merge_sha": merge_sha,
            "spec_hash": spec_hash, "deltas": deltas, "verification": {},
            "error": trigger_delta.get("error"),
        }

    # Phase 6: Post-apply verification
    phase("verify")
    verification = await verify_post_apply(lock, workflow_id, session_id)

    # A failed (or unperformable) post-apply check FAILS the activation.
    # Collecting a false and returning success would report a broken lane as live.
    bad = failed_checks(verification)
    if bad:
        error = "post-apply verification failed: " + ", ".join(bad)
        log(f"lane_activate FAILED verification: {error}")
        return {
            "outcome": "failed", "lane": lane, "merge_sha": merge_sha,
            "spec_hash": spec_hash, "deltas": deltas, "verification": verification,
            "error": error,
        }

    # Determine outcome
    any_changed = any(d.get("action") in ("created", "updated") for d in deltas)
    outcome = "activated" if any_changed else "no_op"

    result = {
        "outcome": outcome,
        "lane": lane,
        "merge_sha": merge_sha,
        "spec_hash": spec_hash,
        "deltas": deltas,
        "verification": verification,
        "error": None,
    }
    log(f"lane_activate complete: outcome={outcome}")
    return result
'''
