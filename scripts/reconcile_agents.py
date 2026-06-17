#!/usr/bin/env python3
"""IaC-sync v1 (agents): reconcile live aios agents from committed manifests.

The dev-pipeline's opus agents (``dev-implement``/``dev-review``/``dev-fix``/
``dev-ci-watch``/``dev-risk``) exist ONLY as live DB rows — 100% imperative and
unversioned, so a change to any is a hand-edit no one can diff against intent.
This script brings agents under git, mirroring the workflow reconciler
(``reregister_workflows.py``): committed thin, secret-free manifests under
``infra/agents/*.json`` are the source of truth; the live state is a reconciled
projection.  A master push touching a manifest fires the deploy-on-merge Action
(``.github/workflows/reconcile-agents.yml``), which runs this script to read each
live agent BY NAME, diff it against the manifest, and PUT a version-bumped update
in place — idempotent, fail-loud.

See ``infra/agents/README.md`` for the manifest shape, the seed-extraction
bootstrap, and the load-bearing id-vs-name constraint (the dev-pipeline workflow
references each agent by its server-minted ``agent_<ULID>`` id; this reconciler
reconciles by the human-stable ``name``).  Those compose ONLY on the UPDATE path:
a CREATE mints a NEW id the referencing workflow header does not point at yet, so
create is bootstrap/disaster-recovery only and emits a loud rewire warning.

Design notes
------------
* Split into PURE helpers (``load_manifests`` / ``desired_diff``) that are offline
  unit-testable, plus a thin HTTP shell (``find_live_agent`` / ``reconcile_agent``)
  that touches the network.  This mirrors ``reregister_workflows.py``.
* Manifests are validated against the REAL ``AgentCreate`` (``extra='forbid'``) at
  load time, so a malformed/extra-field/secret-shaped manifest fails the build,
  not prod.
* Diffing normalises both sides through ``AgentCreate(...).model_dump()`` so nested
  default-population / field-ordering differences (tools/skills/mcp_servers/
  http_servers are lists of dicts) do not produce phantom drift — we compare the
  canonical serialised shape, never raw manifest text.
* Fail-loud: any per-manifest failure aborts non-zero.  ``find_live_agent`` raises
  on >=2 name matches (never silently picks one — prod already has near-duplicate
  names).  A 409 on PUT is fatal.  ``--check`` performs NO write and exits
  non-zero iff any drift is found (the seed of the v2 ops-agent drift loop).
* Config (``AIOS_URL`` + ``AIOS_API_KEY``) is read from the environment ONLY,
  NEVER argv — an argv-borne secret leaks via ``ps`` / ``/proc/<pid>/cmdline``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from aios.models.agents import AgentCreate as _AgentCreate

DEFAULT_MANIFEST_DIR = "infra/agents"

# Hard ceiling on any single HTTP call so a hung connection fails fast instead of
# stalling until the GitHub job's own timeout.
_REQUEST_TIMEOUT_S = 30

Verdict = Literal["create", "update", "noop"]


class ReconcileError(RuntimeError):
    """A loud, fatal failure — abort the write / fail the job."""


# ─── pure helpers (offline-testable, no network) ─────────────────────────────


def _agent_create_cls() -> type[_AgentCreate]:
    """Import ``AgentCreate`` lazily so the pure-import surface of this module stays
    small and a missing dep surfaces loudly at run, mirroring the workflow script's
    lazy builder import."""
    from aios.models.agents import AgentCreate

    return AgentCreate


def load_manifests(manifest_dir: str | Path) -> list[dict[str, Any]]:
    """Read every ``*.json`` under ``manifest_dir``, parse, and validate each against
    the real ``AgentCreate`` (``extra='forbid'``) so a malformed / extra-field /
    secret-shaped manifest fails HERE (the build), not prod.

    Returns the list of raw manifest dicts (sorted by filename for stable ordering).
    Raises ``ReconcileError`` on a bad path, malformed JSON, or a validation failure.
    """
    AgentCreate = _agent_create_cls()
    directory = Path(manifest_dir)
    if not directory.is_dir():
        raise ReconcileError(f"manifest dir does not exist: {directory}")

    manifests: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ReconcileError(f"{path.name}: not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ReconcileError(f"{path.name}: manifest must be a JSON object")
        try:
            AgentCreate(**raw)
        except Exception as exc:  # pydantic ValidationError or similar
            raise ReconcileError(
                f"{path.name}: does not validate against AgentCreate: {exc}"
            ) from exc
        manifests.append(raw)
    return manifests


def _canonical(fields: dict[str, Any]) -> dict[str, Any]:
    """Normalise an agent's versioned fields to the canonical serialised shape.

    Both the manifest and the LIVE read view are routed through the SAME
    ``AgentCreate(...).model_dump()`` so nested default-population and field-ordering
    differences are erased — we compare canonical shapes, never raw text."""
    AgentCreate = _agent_create_cls()
    versioned = {k: v for k, v in fields.items() if k in AgentCreate.model_fields}
    return AgentCreate(**versioned).model_dump()


def desired_diff(manifest: dict[str, Any], live: dict[str, Any] | None) -> Verdict:
    """Decide the reconcile verdict for one agent.

    ``None`` live → ``create``; canonical-field-identical → ``noop``; else ``update``.
    Comparison normalises both sides through ``AgentCreate`` so field-order /
    default-population differences in the nested list-of-dict fields (tools / skills /
    mcp_servers / http_servers) do not produce phantom drift.
    """
    if live is None:
        return "create"
    return "noop" if _canonical(manifest) == _canonical(live) else "update"


# ─── HTTP I/O (the thin, network-touching shell) ─────────────────────────────


def _request(
    method: str, url: str, api_key: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        # timeout: a hung connection must fail fast (≈30s) rather than wait out
        # the GitHub job timeout.
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return exc.code, payload
    except urllib.error.URLError as exc:
        # DNS / connection refused / timeout: surface as the clean fatal path
        # (ReconcileError → FATAL:, non-zero exit) instead of an uncaught traceback.
        raise ReconcileError(f"{method} {url} failed: {exc.reason}") from exc


def find_live_agent(base_url: str, name: str, api_key: str) -> dict[str, Any] | None:
    """Look up a live agent BY exact name via ``GET /v1/agents?name=<name>``.

    The endpoint returns a LIST (``_list_scoped``, newest first), NOT a unique row.
    Required behaviour: 0 matches → ``None``; exactly 1 → that dict; **>=2 → raise
    ``ReconcileError`` (fail-loud)** — never silently pick one (prod already has
    near-duplicate names, so this guard is load-bearing).
    """
    url = f"{base_url.rstrip('/')}/v1/agents?name={urllib.parse.quote(name)}"
    status, payload = _request("GET", url, api_key)
    if not (200 <= status < 300):
        raise ReconcileError(
            f"GET agents?name={name} returned {status}: {json.dumps(payload)[:500]}"
        )
    items = payload.get("items")
    if not isinstance(items, list):
        raise ReconcileError(
            f"GET agents?name={name} response missing list 'items': {json.dumps(payload)[:500]}"
        )
    # Defensive: the equality filter SHOULD return only exact matches, but a future
    # endpoint regression to a prefix/substring match would silently mis-reconcile,
    # so re-assert exact-name equality here before trusting the rows.
    exact = [a for a in items if isinstance(a, dict) and a.get("name") == name]
    if len(exact) == 0:
        return None
    if len(exact) >= 2:
        ids = ", ".join(str(a.get("id")) for a in exact)
        raise ReconcileError(
            f"found {len(exact)} live agents named {name!r} (ids: {ids}) — refusing to guess "
            "which to reconcile; resolve the duplicate names first"
        )
    return exact[0]


def reconcile_agent(
    manifest: dict[str, Any], *, base_url: str, api_key: str, check: bool
) -> Verdict:
    """Reconcile ONE agent manifest: find live BY NAME → diff → decide → write.

    On ``create``: ``POST /v1/agents`` AND emit the loud rewire warning (see module
    docstring + README — a create mints a new id the referencing workflow header
    does not point at yet).  On ``update``: read live ``version`` and PUT the
    manifest fields with that version (409 → fail loud).  On ``noop``: nothing.

    When ``check=True``: compute the verdict, print it, and perform NO write (no
    POST, no PUT).  Returns the verdict either way.
    """
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        raise ReconcileError(f"manifest is missing a string 'name': {json.dumps(manifest)[:300]}")

    live = find_live_agent(base_url, name, api_key)
    verdict = desired_diff(manifest, live)

    if check:
        label = {"create": "WOULD-CREATE", "update": "WOULD-UPDATE", "noop": "in-sync"}[verdict]
        print(f"[{name}] {label}")
        if verdict == "create":
            # Surface the rewire caveat even in check mode so a drift report makes the
            # bootstrap-only nature of create explicit.
            print(
                f"WARNING: agent {name} does not exist live; a create would mint a NEW id that "
                "the referencing workflow header does not point at — manual rewire required."
            )
        return verdict

    if verdict == "noop":
        print(f"[{name}] in-sync — no-op (no version bump).")
        return verdict

    if verdict == "create":
        created = _post_agent(base_url, api_key, manifest)
        new_id = created.get("id")
        # The loud create-warning mandated by the CRITICAL DESIGN CONSTRAINT: a fresh
        # agent is live but the dev-pipeline workflow header still points at the PRIOR
        # id, so the pipeline cannot reach it until the workflow is re-registered.
        print(
            f"WARNING: created agent {name} id={new_id}; the referencing workflow header "
            "still points at the prior id and must be re-registered to rewire."
        )
        return verdict

    # update — verdict=="update" implies live was found (desired_diff returns
    # "create" when live is None), so this narrowing always holds.
    assert live is not None
    live_version = live.get("version")
    agent_id = live.get("id")
    if live_version is None or not isinstance(agent_id, str):
        raise ReconcileError(
            f"[{name}] live agent response missing 'id'/'version': {json.dumps(live)[:500]}"
        )
    print(f"[{name}] drift detected — updating (PUT id={agent_id} with version={live_version})…")
    updated = _put_agent(base_url, agent_id, api_key, version=live_version, manifest=manifest)
    new_version = updated.get("version")
    if new_version is None or new_version <= live_version:
        raise ReconcileError(
            f"[{name}] PUT succeeded but version did not increment "
            f"(live={live_version}, response={new_version})"
        )
    print(f"[{name}] updated: version {live_version} → {new_version}.")
    return verdict


def _versioned_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    AgentCreate = _agent_create_cls()
    return {k: v for k, v in manifest.items() if k in AgentCreate.model_fields}


def _post_agent(base_url: str, api_key: str, manifest: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/agents"
    status, payload = _request("POST", url, api_key, body=_versioned_fields(manifest))
    if not (200 <= status < 300):
        raise ReconcileError(f"POST agent returned {status}: {json.dumps(payload)[:500]}")
    return payload


def _put_agent(
    base_url: str, agent_id: str, api_key: str, *, version: int, manifest: dict[str, Any]
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/agents/{agent_id}"
    body = {"version": version, **_versioned_fields(manifest)}
    status, payload = _request("PUT", url, api_key, body=body)
    if status == 409:
        raise ReconcileError(
            "PUT returned 409 version-mismatch — the live agent was bumped concurrently "
            f"(version {version} is stale). Re-run the job. Body: {json.dumps(payload)[:500]}"
        )
    if not (200 <= status < 300):
        raise ReconcileError(f"PUT agent returned {status}: {json.dumps(payload)[:500]}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Config is read from the environment ONLY. In particular AIOS_API_KEY is NEVER a
    # CLI flag: an argv-borne secret is exposed via `ps` / /proc/<pid>/cmdline. The
    # Action passes it solely via `env:`.
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "read-only diff mode: print each agent's verdict (in-sync / WOULD-CREATE / "
            "WOULD-UPDATE), perform NO write, and exit non-zero iff any drift is found. "
            "This is the seed of the v2 ops-agent drift loop."
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        default=DEFAULT_MANIFEST_DIR,
        metavar="DIR",
        help=f"directory of *.json agent manifests (default: {DEFAULT_MANIFEST_DIR}).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="reconcile only the named agent(s) (matched on manifest 'name'). Repeatable.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("AIOS_API_KEY")
    url = os.environ.get("AIOS_URL")
    if not api_key:
        print("FATAL: AIOS_API_KEY is not set", file=sys.stderr)
        return 2
    if not url:
        print("FATAL: AIOS_URL is not set", file=sys.stderr)
        return 2

    try:
        manifests = load_manifests(args.manifest_dir)
    except ReconcileError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {m.get("name") for m in manifests}
        if unknown:
            print(f"FATAL: unknown --only agent(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        manifests = [m for m in manifests if m.get("name") in wanted]

    if not manifests:
        print(f"FATAL: no agent manifests found under {args.manifest_dir}", file=sys.stderr)
        return 2

    counts = {"create": 0, "update": 0, "noop": 0}
    for manifest in manifests:
        try:
            verdict = reconcile_agent(manifest, base_url=url, api_key=api_key, check=args.check)
        except ReconcileError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 1
        counts[verdict] += 1

    print(f"Done: {counts['create']} created, {counts['update']} updated, {counts['noop']} noop.")

    if args.check and (counts["create"] or counts["update"]):
        # Drift detected under --check: exit non-zero so the scheduled v2 check fails
        # loudly (live ≠ declared), but write nothing.
        print(
            f"DRIFT: {counts['create']} would-create, {counts['update']} would-update "
            "(no writes performed under --check).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
