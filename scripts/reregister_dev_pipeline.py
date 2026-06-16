#!/usr/bin/env python3
"""IaC-sync v1: re-register the live dev-pipeline workflow from its builder.

When ``src/aios/workflows/dev_pipeline.py`` changes on ``master``, the merged fix
to ``build_dev_pipeline_script()`` is INERT — a run snapshots ``workflows.script``
at launch and there is no startup re-sync, so the live workflow object keeps
executing the OLD script until someone manually re-registers it (proven live
2026-06-16: #1177 merged but every run ran the old script until a hand
re-register).  This script automates the hand recipe that activated #1177+#1159
(v2→v3): read the live workflow, re-extract its configurable constants, rebuild
the script from the *updated* builder, validate it, and PUT it back — bumping the
version — so merged builder fixes take effect automatically.

This is **IaC-sync v1** for the workflow object: a one-shot reconcile of a single
known object, driven by a deploy-on-merge GitHub Action.  The v2 evolution — a
reconcile loop over manifests for all infra (agents/triggers/envs as source of
truth) — is sketched in eumemic-company ``architecture/dev-machine-reification.md``
§ infrastructure-as-code.  See ``.github/workflows/reregister-dev-pipeline.yml``.

Design notes
------------
* Constants are read from the LIVE script header (the ``NAME = <repr>`` lines),
  NOT hardcoded — that is what keeps the re-register drift-safe across prod config
  changes (a new agent id / sentinel set / merge tier survives the round-trip).
* The script is split into pure functions (extract / regenerate / validate /
  decide) and a thin ``main`` that does the HTTP I/O, so the load-bearing logic is
  unit-testable offline.
* It is idempotent: when the regenerated script equals the live one, it no-ops and
  exits 0 with no version bump.  It fails LOUDLY (non-zero exit) on a validation
  failure, a 409 version-mismatch, or any non-2xx — drift must be visible, never
  silently skipped.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

# The configurable constants the builder accepts, in the order they appear in the
# rendered header.  Each maps a header constant NAME (read from the LIVE script) to
# the ``build_dev_pipeline_script`` keyword it feeds.  Reading these from the
# deployed workflow (rather than hardcoding) is what preserves prod config across
# the re-register.  Constants beyond these (MIN_SPEC_BODY_WORDS, the *_SCHEMA dicts,
# …) are NOT builder kwargs — they are emitted by the builder from module-level
# defaults, so re-deriving them from the rebuilt script would be circular; they are
# intentionally excluded here.
CONSTANT_TO_KWARG: dict[str, str] = {
    "IMPLEMENT_AGENT_ID": "implement_agent_id",
    "REVIEW_AGENT_ID": "review_agent_id",
    "FIX_AGENT_ID": "fix_agent_id",
    "CI_AGENT_ID": "ci_agent_id",
    "RISK_AGENT_ID": "risk_agent_id",
    "GITHUB_SERVER": "github_server",
    "BASE_BRANCH": "base_branch",
    "MERGE_SENTINELS": "merge_sentinels",
    "MAX_REVIEW_ITERS": "max_review_iters",
    "MAX_CI_ITERS": "max_ci_iters",
    "AUTO_MERGE_MAX_TIER": "auto_merge_max_tier",
    "MERGE_METHOD": "merge_method",
    "DEFAULT_MODEL": "default_model",
}


class ReregisterError(RuntimeError):
    """A loud, fatal failure — abort the PUT / fail the job."""


def extract_constants(live_script: str) -> dict[str, Any]:
    """Pull the builder-configurable constants out of the LIVE script header.

    The header is a flat sequence of ``NAME = <repr>`` assignments emitted by
    ``_render_constants``.  We parse the module with ``ast`` and read the literal
    value of each top-level assignment whose target is one of the known builder
    constants — so the extraction is robust to header reordering and to whatever
    config the live deployment carries.  Raises if any expected constant is
    missing (a header that no longer carries a constant means we'd silently
    re-deploy with a builder default, which is exactly the drift we're guarding).
    """
    try:
        module = ast.parse(live_script)
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ReregisterError(f"live script failed to parse: {exc}") from exc

    found: dict[str, Any] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name not in CONSTANT_TO_KWARG:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                raise ReregisterError(
                    f"could not literal-eval header constant {name}: {exc}"
                ) from exc
            found[name] = value

    missing = sorted(set(CONSTANT_TO_KWARG) - set(found))
    if missing:
        raise ReregisterError(
            "live script header is missing expected constants: " + ", ".join(missing)
        )

    return {CONSTANT_TO_KWARG[name]: value for name, value in found.items()}


def regenerate_script(constants: dict[str, Any]) -> str:
    """Rebuild the production script from the UPDATED builder + live constants."""
    # Imported here (not at module top) so the unit tests that exercise the pure
    # extract/decide helpers do not require the full aios import surface, and so a
    # missing dependency surfaces as a loud ImportError inside the run, not at
    # collection time.
    from aios.workflows.dev_pipeline import build_dev_pipeline_script

    return build_dev_pipeline_script(**constants)


def validate_script(new_script: str) -> None:
    """Fail closed if the regenerated script is not even valid Python.

    ``ast.parse`` is the minimum gate run inline here; the Action ALSO runs
    ``tests/integration/test_wf_dev_pipeline_fixture.py`` (which execs the emitted
    program) as a separate, harder gate before invoking this script's PUT path.
    """
    try:
        ast.parse(new_script)
    except SyntaxError as exc:
        raise ReregisterError(f"regenerated script is not valid Python: {exc}") from exc


def needs_update(new_script: str, live_script: str) -> bool:
    """Idempotency check: only PUT when the regenerated script actually differs."""
    return new_script != live_script


# ─── HTTP I/O (the thin, network-touching shell) ─────────────────────────────

# Hard ceiling on any single HTTP call so a hung connection fails fast instead of
# stalling until the GitHub job's own timeout.
_REQUEST_TIMEOUT_S = 30


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
        # (ReregisterError → FATAL:, non-zero exit) instead of an uncaught
        # traceback. (HTTPError is a URLError subclass but is handled above.)
        raise ReregisterError(f"{method} {url} failed: {exc.reason}") from exc


def get_workflow(base_url: str, workflow_id: str, api_key: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/workflows/{workflow_id}"
    status, payload = _request("GET", url, api_key)
    if not (200 <= status < 300):
        raise ReregisterError(f"GET workflow returned {status}: {json.dumps(payload)[:500]}")
    return payload


def put_workflow(
    base_url: str, workflow_id: str, api_key: str, *, version: int, script: str
) -> dict[str, Any]:
    """PUT the new script. Omitted tools/http_servers are PRESERVED (operator path)."""
    url = f"{base_url.rstrip('/')}/v1/workflows/{workflow_id}"
    status, payload = _request("PUT", url, api_key, body={"version": version, "script": script})
    if status == 409:
        raise ReregisterError(
            "PUT returned 409 version-mismatch — the live workflow was bumped "
            "concurrently (version "
            f"{version} is stale). Re-run the job. Body: {json.dumps(payload)[:500]}"
        )
    if not (200 <= status < 300):
        raise ReregisterError(f"PUT workflow returned {status}: {json.dumps(payload)[:500]}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Config is read from the environment only. In particular AIOS_API_KEY is
    # NEVER a CLI flag: an argv-borne secret is exposed via `ps` /
    # /proc/<pid>/cmdline on a manual run. The Action passes it solely via `env:`.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "log the extracted live workflow constants (agent ids, sentinels, "
            "tiers, model). Off by default to avoid writing prod config to CI logs."
        ),
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("AIOS_API_KEY")
    url = os.environ.get("AIOS_URL")
    workflow_id = os.environ.get("DEV_PIPELINE_WORKFLOW_ID")

    if not api_key:
        print("FATAL: AIOS_API_KEY is not set", file=sys.stderr)
        return 2
    if not url:
        print("FATAL: AIOS_URL is not set", file=sys.stderr)
        return 2
    if not workflow_id:
        print("FATAL: DEV_PIPELINE_WORKFLOW_ID is not set", file=sys.stderr)
        return 2

    try:
        # 1. Read the live workflow → current version + script.
        live = get_workflow(url, workflow_id, api_key)
        live_script = live.get("script")
        live_version = live.get("version")
        if not isinstance(live_script, str) or live_version is None:
            raise ReregisterError(
                f"live workflow response missing 'script'/'version': {json.dumps(live)[:500]}"
            )
        print(f"Live dev-pipeline workflow version={live_version}")

        # 2. Extract configurable constants from the LIVE header (drift-safe).
        constants = extract_constants(live_script)
        # The constants are prod workflow config (agent ids, sentinels, tiers,
        # model). Not secret, but don't dump them to CI logs by default — only
        # under --verbose for local debugging.
        if args.verbose:
            print(
                "Extracted live constants: "
                + json.dumps({k: constants[k] for k in sorted(constants)})
            )
        else:
            print(f"Extracted {len(constants)} live constants from the workflow header.")

        # 3. Rebuild from the UPDATED builder.
        new_script = regenerate_script(constants)

        # 4. Validate before any write (fail closed).
        validate_script(new_script)

        # 5. Idempotency: no-op when regenerated == live.
        if not needs_update(new_script, live_script):
            print(
                "Regenerated script is identical to the live script — no-op "
                "(no version bump). IaC is in sync."
            )
            return 0

        # 6. PUT (version bumps; omitted tools/http_servers preserved).
        print(f"Script changed — re-registering (PUT with version={live_version})…")
        updated = put_workflow(
            url,
            workflow_id,
            api_key,
            version=live_version,
            script=new_script,
        )

        # 7. Verify the version actually incremented.
        new_version = updated.get("version")
        if new_version is None or new_version <= live_version:
            raise ReregisterError(
                f"PUT succeeded but version did not increment "
                f"(live={live_version}, response={new_version})"
            )
        print(f"Re-registered dev-pipeline workflow: version {live_version} → {new_version}.")
        return 0

    except ReregisterError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
