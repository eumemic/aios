"""Minimal stdlib AIOS API client for the WaM fusion eval harness.

Zero third-party deps (urllib only) so the harness runs anywhere with
``AIOS_URL`` + ``AIOS_API_KEY`` in the environment. Deliberately NOT the
generated SDK: the eval is a black-box prod-integration probe and must not be
coupled to in-repo SDK versions.

The API key is read from ``$AIOS_API_KEY`` and never logged. Pass the test
account's key (an *operator* principal over its own account — the HTTP path that
the ``workflow:`` model-binding privilege permits).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class AiosError(RuntimeError):
    """An AIOS API call returned a non-2xx status."""


class AiosClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ["AIOS_URL"]).rstrip("/")
        self._api_key = api_key or os.environ["AIOS_API_KEY"]
        if not self._api_key:
            raise AiosError("AIOS_API_KEY is empty")

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:500]
            raise AiosError(f"{method} {path} -> {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else None

    # ── resources ────────────────────────────────────────────────────────────
    def whoami(self) -> dict[str, Any]:
        return self._request("GET", "/v1/accounts/me")

    def create_workflow(self, name: str, script: str, description: str = "") -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/workflows",
            {"name": name, "script": script, "description": description},
        )

    def ensure_workflow(self, name: str, script: str, description: str = "") -> dict[str, Any]:
        """Create the workflow, or reuse + UPDATE-to-current the one already named ``name``.

        Re-registration discipline: a stale workflow body must never score the new
        eval. If a workflow with this name exists, we PUT the current script so its
        version is bumped (no-op if byte-identical) — the returned id+version is the
        one to bind. A bound ``workflow:<id>`` floats to current; binding
        ``workflow:<id>@<version>`` pins the exact version this run registered.
        """
        try:
            return self.create_workflow(name, script, description)
        except AiosError as exc:
            if "409" not in str(exc):
                raise
            resp = self._request("GET", f"/v1/workflows?name={name}&limit=10")
            items = resp.get("data", resp) if isinstance(resp, dict) else resp
            existing = next((w for w in (items or []) if w.get("name") == name), None)
            if existing is None:
                raise
            updated = self._request(
                "PUT",
                f"/v1/workflows/{existing['id']}",
                {"version": existing["version"], "script": script, "description": description},
            )
            return updated

    def create_agent(
        self,
        name: str,
        model: str,
        system: str,
        litellm_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "model": model, "system": system, "tools": []}
        if litellm_extra is not None:
            body["litellm_extra"] = litellm_extra
        return self._request("POST", "/v1/agents", body)

    def create_environment(self, name: str) -> dict[str, Any]:
        return self._request("POST", "/v1/environments", {"name": name})

    def ensure_environment(self, name: str) -> dict[str, Any]:
        """Reuse an environment with ``name`` if present, else create it (idempotent)."""
        try:
            return self.create_environment(name)
        except AiosError as exc:
            if "409" not in str(exc):
                raise
            resp = self._request("GET", "/v1/environments?limit=100")
            items = resp.get("data", resp) if isinstance(resp, dict) else resp
            for env in items or []:
                if env.get("name") == name and env.get("archived_at") is None:
                    return env
            raise

    def create_session(self, agent_id: str, environment_id: str, message: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/sessions",
            {
                "agent_id": agent_id,
                "environment_id": environment_id,
                "initial_message": message,
            },
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/sessions/{session_id}")

    def session_events(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        resp = self._request("GET", f"/v1/sessions/{session_id}/events?limit={limit}")
        return resp.get("data", resp) if isinstance(resp, dict) else resp

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/runs/{run_id}")

    # ── derived helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _event_data(ev: dict[str, Any]) -> dict[str, Any]:
        d = ev.get("data")
        return d if isinstance(d, dict) else ev

    def session_usage(self, session_id: str) -> dict[str, int]:
        """The session's cumulative token usage meter ({input_tokens, output_tokens, ...})."""
        sess = self.get_session(session_id)
        return sess.get("usage") or {}

    def latest_assistant(self, session_id: str) -> dict[str, Any] | None:
        """Return the most recent assistant message event's {content, ...}, or None."""
        events = self.session_events(session_id, limit=100)
        for ev in reversed(events):
            d = self._event_data(ev)
            kind = ev.get("type") or ev.get("kind")
            role = d.get("role") or ev.get("role")
            content = d.get("content") if d.get("content") is not None else ev.get("content")
            if kind == "message" and role == "assistant" and content:
                return {"content": content, "raw": ev}
        return None

    def park_run_id(self, session_id: str) -> str | None:
        """For a workflow-bound session, the inner run id sealed by model_workflow_park."""
        for ev in self.session_events(session_id, limit=100):
            d = self._event_data(ev)
            if d.get("event") == "model_workflow_park":
                return d.get("run_id")
        return None

    def harvest_markers(self, session_id: str) -> list[str]:
        """The ordered model_workflow_* span event names in a session."""
        out = []
        for ev in self.session_events(session_id, limit=100):
            d = self._event_data(ev)
            evt = d.get("event")
            if isinstance(evt, str) and evt.startswith("model_workflow"):
                out.append(evt)
        return out

    def wait_for_assistant(
        self, session_id: str, timeout_s: float = 90.0, poll_s: float = 3.0
    ) -> dict[str, Any] | None:
        """Poll until an assistant turn lands or the session errors / times out."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            asst = self.latest_assistant(session_id)
            if asst is not None:
                return asst
            sess = self.get_session(session_id)
            if sess.get("status") == "errored":
                return {"content": None, "errored": True, "raw": sess}
            time.sleep(poll_s)
        return None
