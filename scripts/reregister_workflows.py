#!/usr/bin/env python3
"""IaC-sync v1: re-register live aios workflows from their builders.

When a workflow builder under ``src/aios/workflows/`` changes on ``master``, the
merged fix is INERT — a run snapshots ``workflows.script`` at launch and there is
no startup re-sync, so the live workflow object keeps executing the OLD script
until someone manually re-registers it (proven live 2026-06-16: dev-pipeline #1177
merged but every run ran the old script until a hand re-register).  This script
automates the hand recipe: for each registered workflow target, read the live
workflow, re-extract its configurable constants from the LIVE header (drift-safe),
rebuild the script from the *updated* builder, validate it, and PUT it back —
bumping the version — so merged builder fixes take effect automatically.

This is **IaC-sync v1** for the workflow objects: a one-shot reconcile of a known
SET of objects, driven by a deploy-on-merge GitHub Action.  Originally this script
re-registered only the dev-pipeline (``reregister_dev_pipeline.py``); aios#1226
GENERALISED it to register a *list* of workflow files so a master push touching any
registered builder (dev-pipeline, triage-pipeline, …) re-registers exactly that
workflow with an auto version bump.  Adding a workflow is one ``WorkflowTarget``
entry in ``TARGETS`` — the extract → regenerate → validate → decide → PUT pipeline
is shared, not forked.  The v2 evolution — a reconcile loop over manifests for all
infra (agents/triggers/envs as source of truth) — is sketched in eumemic-company
``architecture/dev-machine-reification.md`` § infrastructure-as-code.  See
``.github/workflows/reregister-workflows.yml``.

Design notes
------------
* Constants are read from the LIVE script header (the ``NAME = <repr>`` lines),
  NOT hardcoded — that is what keeps the re-register drift-safe across prod config
  changes (a new agent id / sentinel set / merge tier survives the round-trip).
* The script is split into pure functions (extract / regenerate / validate /
  decide) parameterised by a ``WorkflowTarget`` and a thin ``main`` that does the
  HTTP I/O, so the load-bearing logic is unit-testable offline.
* It is idempotent: when a regenerated script equals the live one, that target
  no-ops with no version bump.  It fails LOUDLY (non-zero exit) on a validation
  failure, a 409 version-mismatch, or any non-2xx — drift must be visible, never
  silently skipped.  Each target is independent: a builder a target whose env var
  is unset is SKIPPED (not every deployment registers every workflow), but a
  configured target that fails aborts with a non-zero exit.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ─── the workflow-target registry (generalised in aios#1226) ─────────────────


@dataclass(frozen=True)
class WorkflowTarget:
    """One re-registerable workflow: how to find its live object, rebuild it, and
    map its header constants back to builder kwargs.

    * ``key`` — a short stable identifier used in log lines.
    * ``id_env`` — the environment variable holding the live workflow id (e.g.
      ``DEV_PIPELINE_WORKFLOW_ID``).  A target whose env var is UNSET is skipped
      (a deployment may not register every workflow), so this stays DRY without
      forcing every deployment to configure every target.
    * ``builder`` — the ``build_*_script(**constants) -> str`` callable.
    * ``constant_to_kwarg`` — maps a header constant NAME (read from the LIVE
      script via ast) to the builder keyword it feeds.  Reading these from the
      deployed workflow (not hardcoding) preserves prod config across the
      round-trip; only the builder-configurable constants belong here (the
      *_SCHEMA dicts etc. are builder-default-derived and excluded).
    """

    key: str
    id_env: str
    builder: Callable[..., str]
    constant_to_kwarg: dict[str, str]


# dev-pipeline constant→kwarg map (unchanged from the original single-target script).
DEV_PIPELINE_CONSTANT_TO_KWARG: dict[str, str] = {
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

# triage-pipeline constant→kwarg map (aios#1226).
TRIAGE_PIPELINE_CONSTANT_TO_KWARG: dict[str, str] = {
    "TRIAGE_AGENT_ID": "triage_agent_id",
    "GITHUB_SERVER": "github_server",
    "DEFAULT_REPO": "repo",
    "MAX_ISSUES_PER_RUN": "max_issues_per_run",
    "DEFAULT_MODEL": "default_model",
}


def _dev_pipeline_builder() -> Callable[..., str]:
    # Imported lazily (mirrors the original) so the pure extract/decide tests do not
    # require the full aios import surface, and a missing dep surfaces loudly at run.
    from aios.workflows.dev_pipeline import build_dev_pipeline_script

    return build_dev_pipeline_script


def _triage_pipeline_builder() -> Callable[..., str]:
    from aios.workflows.triage_pipeline import build_triage_pipeline_script

    return build_triage_pipeline_script


def _targets() -> list[WorkflowTarget]:
    """The registered re-registerable workflows. Lazily resolves builders so the pure
    helpers stay importable without the aios surface; ``TARGETS`` below is the eager
    handle the Action and tests use."""
    return [
        WorkflowTarget(
            key="dev-pipeline",
            id_env="DEV_PIPELINE_WORKFLOW_ID",
            builder=_dev_pipeline_builder(),
            constant_to_kwarg=DEV_PIPELINE_CONSTANT_TO_KWARG,
        ),
        WorkflowTarget(
            key="triage-pipeline",
            id_env="TRIAGE_PIPELINE_WORKFLOW_ID",
            builder=_triage_pipeline_builder(),
            constant_to_kwarg=TRIAGE_PIPELINE_CONSTANT_TO_KWARG,
        ),
    ]


class ReregisterError(RuntimeError):
    """A loud, fatal failure — abort the PUT / fail the job."""


def extract_constants(live_script: str, constant_to_kwarg: dict[str, str]) -> dict[str, Any]:
    """Pull the builder-configurable constants out of the LIVE script header.

    The header is a flat sequence of ``NAME = <repr>`` assignments emitted by the
    builder's ``_render_constants``.  We parse the module with ``ast`` and read the
    literal value of each top-level assignment whose target is one of the known
    builder constants — so extraction is robust to header reordering and to whatever
    config the live deployment carries.  Raises if any expected constant is missing
    (a header that no longer carries a constant means we'd silently re-deploy with a
    builder default, which is exactly the drift we're guarding).
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
            if name not in constant_to_kwarg:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                raise ReregisterError(
                    f"could not literal-eval header constant {name}: {exc}"
                ) from exc
            found[name] = value

    missing = sorted(set(constant_to_kwarg) - set(found))
    if missing:
        raise ReregisterError(
            "live script header is missing expected constants: " + ", ".join(missing)
        )

    return {constant_to_kwarg[name]: value for name, value in found.items()}


def regenerate_script(constants: dict[str, Any], builder: Callable[..., str]) -> str:
    """Rebuild the production script from the UPDATED builder + live constants."""
    return builder(**constants)


def validate_script(new_script: str) -> None:
    """Fail closed if the regenerated script is not even valid Python.

    ``ast.parse`` is the minimum gate run inline here; the Action ALSO runs the
    fixture-drive tests (which exec the emitted program) as a separate, harder gate
    before invoking this script's PUT path.
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


def reconcile_target(
    target: WorkflowTarget,
    *,
    base_url: str,
    workflow_id: str,
    api_key: str,
    verbose: bool,
    check: bool = False,
) -> bool:
    """Reconcile ONE workflow target: read live → extract → rebuild → validate →
    decide → PUT.  Returns True if the target CHANGED (a PUT was applied, or — under
    ``check=True`` — a PUT WOULD be applied), False on a no-op (already in sync).
    Raises ReregisterError on any fatal problem.

    When ``check=True`` this runs steps 1-5 (read live → extract → rebuild →
    validate → decide) but SKIPS the PUT (step 6): it prints the per-target verdict
    (``in-sync`` / ``WOULD-RE-REGISTER``) and performs no write. ``main`` then exits
    non-zero iff any target would change — the read-only drift detector that seeds
    the v2 ops-agent Vaughan check.

    This is the shared per-target pipeline — adding a workflow is a TARGETS entry,
    not a fork of this logic."""
    # 1. Read the live workflow → current version + script.
    live = get_workflow(base_url, workflow_id, api_key)
    live_script = live.get("script")
    live_version = live.get("version")
    if not isinstance(live_script, str) or live_version is None:
        raise ReregisterError(
            f"[{target.key}] live workflow response missing 'script'/'version': "
            f"{json.dumps(live)[:500]}"
        )
    print(f"[{target.key}] live workflow version={live_version}")

    # 2. Extract configurable constants from the LIVE header (drift-safe).
    constants = extract_constants(live_script, target.constant_to_kwarg)
    if verbose:
        print(
            f"[{target.key}] extracted live constants: "
            + json.dumps({k: constants[k] for k in sorted(constants)})
        )
    else:
        print(f"[{target.key}] extracted {len(constants)} live constants from the header.")

    # 3. Rebuild from the UPDATED builder.
    new_script = regenerate_script(constants, target.builder)

    # 4. Validate before any write (fail closed).
    validate_script(new_script)

    # 5. Idempotency: no-op when regenerated == live.
    if not needs_update(new_script, live_script):
        if check:
            print(f"[{target.key}] in-sync")
        else:
            print(f"[{target.key}] regenerated script is identical — no-op (no version bump).")
        return False

    # 5b. --check: drift exists, but perform NO write. Report and return changed=True.
    if check:
        print(f"[{target.key}] WOULD-RE-REGISTER")
        return True

    # 6. PUT (version bumps; omitted tools/http_servers preserved).
    print(f"[{target.key}] script changed — re-registering (PUT with version={live_version})…")
    updated = put_workflow(base_url, workflow_id, api_key, version=live_version, script=new_script)

    # 7. Verify the version actually incremented.
    new_version = updated.get("version")
    if new_version is None or new_version <= live_version:
        raise ReregisterError(
            f"[{target.key}] PUT succeeded but version did not increment "
            f"(live={live_version}, response={new_version})"
        )
    print(f"[{target.key}] re-registered: version {live_version} → {new_version}.")
    return True


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
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="KEY",
        help=(
            "reconcile only the named target(s) (e.g. --only triage-pipeline). "
            "Repeatable. Default: every target whose *_WORKFLOW_ID env var is set."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "read-only diff mode: read live → extract → rebuild → validate → decide, "
            "print each target's verdict (in-sync / WOULD-RE-REGISTER), perform NO PUT, "
            "and exit non-zero iff any target would change. Seeds the v2 drift loop."
        ),
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

    targets = _targets()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {t.key for t in targets}
        if unknown:
            print(f"FATAL: unknown --only target(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        targets = [t for t in targets if t.key in wanted]

    reconciled = 0
    skipped = 0
    changed = 0
    for target in targets:
        workflow_id = os.environ.get(target.id_env)
        if not workflow_id:
            # A deployment may not register every workflow — a missing id env var is a
            # SKIP, not a failure (keeps the registry DRY without forcing every env to
            # configure every target).
            print(f"[{target.key}] {target.id_env} not set — skipping.")
            skipped += 1
            continue
        try:
            if reconcile_target(
                target,
                base_url=url,
                workflow_id=workflow_id,
                api_key=api_key,
                verbose=args.verbose,
                check=args.check,
            ):
                changed += 1
            reconciled += 1
        except ReregisterError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 1

    if reconciled == 0 and skipped > 0:
        print(
            "FATAL: no workflow targets were configured (every *_WORKFLOW_ID is unset)",
            file=sys.stderr,
        )
        return 2
    if args.check:
        print(
            f"Done (check): {reconciled} target(s) checked, {changed} would re-register, "
            f"{skipped} skipped."
        )
        if changed:
            # Drift detected under --check: exit non-zero (live ≠ declared), no writes.
            print(
                f"DRIFT: {changed} target(s) would re-register (no writes performed under "
                "--check).",
                file=sys.stderr,
            )
            return 1
        return 0
    print(f"Done: {reconciled} target(s) reconciled, {changed} re-registered, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
