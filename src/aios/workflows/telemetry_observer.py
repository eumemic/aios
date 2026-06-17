"""Machine-observer anomaly-scan loop (aios#1326) — the deterministic substrate-different observer.

Plane C, roadmap item 8 of the substrate-different-verdict epic (aios#1330).

This module stands a DETERMINISTIC observer on the one substrate no agent can talk past:
the host's per-run cost/token/iteration/wall-clock journal (``WfRunUsage``, #1324 / PR#1334).
On every terminal run of a watched dev/conveyor/loop workflow a scripted (NOT agent) workflow
re-derives a waste verdict from those host-emitted numbers — uncorrelated with the LLM nodes
by **substrate** (host numbers, not the run's self-report) and by **stage** (post-terminal,
not in-pass). It grades nobody's homework with an LLM: a numeric-threshold predicate has a
universally-correct policy (the script-vs-agent partition doctrine), so the verdict can be a
pure deterministic script. The moment a verdict needs judgment it would have to become an
``agent()`` — which would re-correlate it — so this observer NEVER calls ``agent()``/``gate()``.

HONEST SCOPE (stated up front, mirrors the issue's non-goals): the observer detects ONLY
mechanically-legible pathologies. Semantically-wrong-but-cheap-and-on-budget output (a
Ralph-Wiggum loop producing a plausible diff), design-misallocation, and coverage-selection
blind spots trip NO numeric anomaly and stay chairman/seat-found (the irreducible residue, by
the uncorrelated-immune theorem). v1 ships with EXACTLY ONE live signal — the LEAD
``cost-on-errored`` signal — fully computable from host-emitted ``status`` + ``cost_microusd``.

─── THE SUBSTRATE-AVAILABILITY MATRIX (governs which signals are buildable) ───────────────

Verified against this repo: the read-path ``WfRunUsage`` (``models/workflows.py``) surfaces
ONLY ``cost_microusd · input_tokens · output_tokens · cache_read_input_tokens ·
cache_creation_input_tokens · iteration_count (ALWAYS null today) · wall_clock_ms``. Of the
four candidate signals, only the LEAD ``cost-on-errored`` is fully computable:

  * LEAD cost-on-errored  — ``status`` + ``cost_microusd``  → BOTH host-emitted → **LIVE**.
  * byte-buffer           — needs ``net_diff_lines``, NOT on the host read path → DEFERRED.
  * cost-on-no-PR         — needs PR-presence, lives only in the run's ``output`` → DEFERRED.
  * iteration-at-max      — ``iteration_count`` is ALWAYS null today → DEFERRED.
  * cost-per-net-line     — needs ``net_diff_lines`` (with byte-buffer) → DEFERRED.

The three DEFERRED signals are specified as frozen-band constants and fixture-locked against a
fabricated ``get_run`` return to PIN their predicate contract, but they are NOT wired into the
live verdict path — an implement agent must not fabricate their substrate fields. They are
gated behind a NEW host primitive (``net_diff_lines`` / PR-presence on the read path) and a
NEW per-run iteration counter — filed as follow-up dependencies.

Why DEFERRED can't masquerade as LIVE: the observer's fail-loud rule (criterion 9) makes ANY
null/missing field used by a FIRED predicate yield ``cannot-determine`` — so a null
``iteration_count`` / absent ``net_diff_lines`` consumed by its predicate can only ever read
``cannot-determine``, never a false ``ok``/``anomaly``.

─── THE INPUT ENVELOPE (TRACED, not assumed from a cited precedent) ───────────────────────

The issue WARNS that an earlier draft falsely cited ``input["trigger"]["event"]["run_id"]`` as
precedent. Tracing the actual construction site (``harness/trigger_runner.py`` —
``compose_workflow_run_input`` + ``_run_workflow``) confirms:

  * A ``WorkflowAction`` fire's run input is ALWAYS the envelope
    ``{"trigger": <firing context>, "input": <input_template verbatim>}``.
  * For a ``run_completion`` fire the completing run rides BY VALUE under
    ``input["trigger"]["run"]`` = ``{id, workflow_id, status, output, error}``
    (``trigger_runner.py``: ``trigger["run"] = {...}``).
  * The ``event`` key is passed as ``None`` for ``run_completion`` fires
    (``event=event if trigger_context == "external_event" else None``) — so there is
    **NO** ``input["trigger"]["event"]`` for run-completion fires. The earlier citation was
    indeed false; the correct key path is ``input["trigger"]["run"]["id"]``.

The observer's ``_unwrap`` therefore reads ``input["trigger"]["run"]["id"]`` for the real
trigger path, and ALSO accepts a bare ``{"run_id": ...}`` (the fixture / arm-time path).
CRITICAL: the observer reads ``trigger.run`` ONLY for the run id — it NEVER reads
``trigger.run.output`` (the run's self-report prose), which is the substrate it is forbidden
to touch. The waste verdict is computed from ``get_run``'s host numbers, never from the
envelope's ``output``.

─── DESIGN (two scripted workflows + the builder idiom) ───────────────────────────────────

Authored with the EXACT ``dev_pipeline.py`` / ``triage_pipeline.py`` builder idiom: an
exported ``build_observer_script(...) -> str`` returning workflow SOURCE (a prepended
frozen-band constants header + a static ``_OBSERVER_BODY`` of pure-stdlib ``re``/``json``,
value-domain I/O, deterministic emit order → replay-stable), plus ``REQUIRED_TOOLS``, and
``build_observer_workflow_create(...) -> WorkflowCreate`` bundling script+surface so the
declared surface can't drift from the script (#1135). The dead-man twin (``observer_sweep``)
is authored identically.

Frozen bands are PREPENDED constants (committed in the loop blueprint, NEVER inferred at
runtime — the body imports neither ``datetime`` nor ``time``); the verdict gates on a frozen
threshold, never on recency.

The observer's tool surface is ``REQUIRED_TOOLS = [get_run, list_runs]`` — no ``http_request``,
no ``bash``, no ``edit``. The observer mutates nothing and touches no external artifact; its
only output is the structured verdict record (a ledger row). This is structural enforcement of
the issue-emission gating (criterion 11): with no ``http_request`` in surface, the observer
CANNOT file a GitHub issue / obliging action-item — so it can't feed the
Class-closed:NO graveyard before the fractal-retro closure machinery (eumemic-company#24) lands.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import ToolSpec
from aios.models.workflows import WorkflowCreate

# ─── frozen bands (git-versioned blueprint constants; bumped together with bands_version) ──
#
# These live as constants in the prepended header and are committed in the loop blueprint.
# The body NEVER infers a band at runtime (no datetime/time import) — it gates on a frozen
# threshold. Bumping any band MUST bump BANDS_VERSION so a stored verdict row records which
# band set produced it (the residue table #1328 keys on it). The DEFERRED-signal bands are
# carried so their fixture contract is pinned; they are NOT consulted by the live verdict.

# LEAD signal [LIVE]: spend (microusd) burned on a run that errored out. A run that errored
# below this floor is benign (a cheap fast-fail). Tuned to the watched-workflow cost profile.
DEFAULT_COST_WASTE_FLOOR = 500_000  # 0.50 USD in micro-USD

# DEFERRED [gated on a real per-run iteration counter; iteration_count is ALWAYS null today].
DEFAULT_ITER_MAX_BAND = 50

# DEFERRED [gated on a host primitive emitting net_diff_lines onto the read path].
DEFAULT_BYTE_BUFFER_INPUT_BAND = 1_000_000  # input_tokens ceiling for the byte-buffer signal
DEFAULT_DIFF_FLOOR = 3  # net_diff_lines below this = "near-zero net diff"

# Bump on ANY band change (the residue table keys verdicts on it).
DEFAULT_BANDS_VERSION = "v1-lead-only-2026-06-17"

# The verdict vocabulary — EXACTLY these three (criterion 3).
VERDICTS: tuple[str, ...] = ("ok", "anomaly", "cannot-determine")

# The one LIVE signal name (criterion 4). The DEFERRED names are carried for documentation /
# the fixture contract only; they never appear in a live run's ``signals_fired``.
LIVE_SIGNALS: tuple[str, ...] = ("cost-on-errored",)
DEFERRED_SIGNALS: tuple[str, ...] = (
    "iteration-at-max",
    "byte-buffer",
    "cost-per-net-accepted-line",
    "cost-on-no-PR",
)


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header (mirrors dev/triage)."""
    return f"{name} = {value!r}"


def _render_observer_constants(
    *,
    cost_waste_floor: int,
    iter_max_band: int,
    byte_buffer_input_band: int,
    diff_floor: int,
    bands_version: str,
) -> str:
    lines = [
        _py("COST_WASTE_FLOOR", cost_waste_floor),
        _py("ITER_MAX_BAND", iter_max_band),
        _py("BYTE_BUFFER_INPUT_BAND", byte_buffer_input_band),
        _py("DIFF_FLOOR", diff_floor),
        _py("BANDS_VERSION", bands_version),
        _py("LIVE_SIGNALS", list(LIVE_SIGNALS)),
        _py("DEFERRED_SIGNALS", list(DEFERRED_SIGNALS)),
    ]
    return "\n".join(lines)


# The static observer body — references the prepended constants. Pure stdlib (re/json),
# value-domain I/O, bounded, NO datetime/time: replay-stable by construction.
_OBSERVER_BODY = r'''
import json


# ─── input-envelope unwrap (TRACED, see module docstring) ─────────────────────

def _run_id(input):
    """Extract the completing run's id from EITHER the bare fixture/arm-time shape
    ``{"run_id": ...}`` OR the real WorkflowAction trigger envelope.

    The real run_completion envelope is
    ``{"trigger": {"id","name","source","fired_at","run": {"id","workflow_id","status",
    "output","error"}}, "input": <template>}`` — TRACED to
    ``harness/trigger_runner.py``'s ``compose_workflow_run_input`` (the completing run
    rides BY VALUE under ``trigger.run``; ``event`` is None for run_completion fires, so
    there is NO ``trigger.event``). We read ONLY ``trigger.run.id`` — never
    ``trigger.run.output`` (the run's self-report prose, the forbidden substrate).

    Returns the run id, or None if no run id is reachably present (a finding, surfaced as
    ``cannot-determine`` by the caller — never papered over).
    """
    if not isinstance(input, dict):
        return None
    # Bare fixture/arm-time path.
    rid = input.get("run_id")
    if isinstance(rid, str) and rid:
        return rid
    # Real trigger envelope.
    trig = input.get("trigger")
    if isinstance(trig, dict):
        run = trig.get("run")
        if isinstance(run, dict):
            rid = run.get("id")
            if isinstance(rid, str) and rid:
                return rid
    return None


# ─── telemetry read (the host substrate; NEVER the run's output/prose) ────────

async def _get_run(run_id):
    """Read the completing run via the get_run builtin. The tool result is the full
    WfRun dict (incl. the ``usage`` object once PR#1334 is deployed) OR an error value
    ``{"error": ...}`` (a NotFound / scope failure resolves as a value, never a raise)."""
    return await tool("get_run", {"run_id": run_id})


def _usage_field(run, name):
    """Pull one WfRunUsage field off a get_run return, distinguishing THREE states the
    observer must never conflate (the load-bearing fail-loud distinction):

    * ('absent',  None) — the ``usage`` object itself is missing/null (a pre-#1334 worker,
      or a degraded read). The observer CANNOT determine spend → cannot-determine.
    * ('null',    None) — ``usage`` is present but this field is explicit null (#1324
      mandates explicit-null not silent-omit; e.g. ``iteration_count`` always-null today,
      or the ``vault_ids:null`` disease shape on ``cost_microusd``). → cannot-determine.
    * ('value',   v)    — a real observed number (note: a real observed ``0`` is a VALUE,
      distinct from null — a childless run sums to a genuine zero).

    Returns ``(state, value)``.
    """
    if not isinstance(run, dict):
        return ("absent", None)
    usage = run.get("usage")
    if not isinstance(usage, dict):
        return ("absent", None)
    if name not in usage:
        # An OMITTED key is the silent-omit disease — treat as cannot-determine, NOT as 0.
        return ("null", None)
    v = usage[name]
    if v is None:
        return ("null", None)
    return ("value", v)


# ─── the deterministic verdict (vs FROZEN bands; fixed signal order) ──────────

def _record(run_id, workflow_id, verdict, signals_fired, snapshot):
    """The structured per-run ledger row — the observer's ONLY output (criterion 3/11).
    Machine-readable telemetry, not a log scrape. No GitHub issue, no obliging action-item
    (structurally impossible: http_request is not in the observer's tool surface)."""
    return {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "verdict": verdict,
        "signals_fired": signals_fired,
        "telemetry_snapshot": snapshot,
        "bands_version": BANDS_VERSION,
    }


def _scan(run):
    """Compute the verdict from host numbers, in fixed order, vs the frozen bands.

    v1 evaluates EXACTLY ONE live signal — the LEAD cost-on-errored signal. The DEFERRED
    signals are NOT evaluated here (they are specified + fixture-locked elsewhere, gated on
    new host primitives). Returns ``(verdict, signals_fired, snapshot)``.

    The verdict is determined by the LEAD predicate alone:
      * status == 'errored' AND cost_microusd > COST_WASTE_FLOOR  → anomaly.
      * status == 'errored' AND cost_microusd is absent/null      → cannot-determine
        (fail-loud: a fired predicate read a missing field — NEVER silently ok).
      * status == 'errored' AND cost_microusd <= COST_WASTE_FLOOR → ok (cheap fast-fail).
      * any non-errored terminal status                          → ok (the lead predicate
        does not fire; no other live signal exists in v1).

    ``workflow_id`` comes off the host read row — NOT the trigger envelope — so the observer
    is stateless w.r.t. which workflow fired it.
    """
    status = run.get("status") if isinstance(run, dict) else None
    workflow_id = run.get("workflow_id") if isinstance(run, dict) else None
    cost_state, cost = _usage_field(run, "cost_microusd")

    # The telemetry snapshot is the host numbers the verdict was computed from (the residue
    # table consumes it). We snapshot the LEAD-relevant fields plus the carried context.
    snapshot = {
        "status": status,
        "cost_microusd": cost if cost_state == "value" else None,
        "cost_microusd_state": cost_state,
    }

    # LEAD signal — cost-on-errored [LIVE]. The predicate FIRES only on an errored run; a
    # field it reads being absent/null is fail-loud cannot-determine, never silent ok.
    if status == "errored":
        if cost_state != "value":
            # A fired predicate consumed a missing/null field → cannot-determine.
            return ("cannot-determine", [], snapshot)
        if cost > COST_WASTE_FLOOR:
            return ("anomaly", ["cost-on-errored"], snapshot)
        return ("ok", [], snapshot)

    # Non-errored terminal: the lead predicate does not fire, and v1 has no other live
    # signal. (A within-bands MERGED run is the eventual ok case for the diff-based signals
    # once they go live — but those are DEFERRED, so any non-errored run reads ok today.)
    if status is None:
        # status itself is missing — we couldn't even read the run row.
        return ("cannot-determine", [], snapshot)
    return ("ok", [], snapshot)


# ─── entry ────────────────────────────────────────────────────────────────────

async def main(input):
    phase("unwrap")
    run_id = _run_id(input)
    if run_id is None:
        # No reachable run id in the envelope — a finding, surfaced loud, not papered over.
        log("observer: no run_id reachable in input envelope")
        return _record(None, None, "cannot-determine", [], {"reason": "no run_id in input"})

    phase("read")
    run = await _get_run(run_id)
    if isinstance(run, dict) and "error" in run and "status" not in run:
        # get_run resolved an error value (NotFound / scope) — cannot read the substrate.
        log("observer: get_run error for", run_id)
        return _record(
            run_id, None, "cannot-determine", [], {"reason": "get_run error"}
        )

    phase("scan")
    verdict, signals_fired, snapshot = _scan(run)
    workflow_id = run.get("workflow_id") if isinstance(run, dict) else None
    log("observer verdict:", verdict, "signals:", signals_fired, "run:", run_id)
    return _record(run_id, workflow_id, verdict, signals_fired, snapshot)
'''


def build_observer_script(
    *,
    cost_waste_floor: int = DEFAULT_COST_WASTE_FLOOR,
    iter_max_band: int = DEFAULT_ITER_MAX_BAND,
    byte_buffer_input_band: int = DEFAULT_BYTE_BUFFER_INPUT_BAND,
    diff_floor: int = DEFAULT_DIFF_FLOOR,
    bands_version: str = DEFAULT_BANDS_VERSION,
) -> str:
    """Return the production ``telemetry_observer`` workflow source.

    Frozen bands are prepended as constants; the body imports neither ``datetime`` nor
    ``time`` and never infers a band at runtime (it gates on the frozen threshold). v1 wires
    EXACTLY ONE live signal — the LEAD cost-on-errored signal — fully computable from
    host-emitted ``status`` + ``cost_microusd``. The other three are specified + fixture-locked
    but DEFERRED behind named host primitives and NOT wired live.
    """
    header = _render_observer_constants(
        cost_waste_floor=cost_waste_floor,
        iter_max_band=iter_max_band,
        byte_buffer_input_band=byte_buffer_input_band,
        diff_floor=diff_floor,
        bands_version=bands_version,
    )
    return header + "\n" + _OBSERVER_BODY


def build_observer_fixture_script(
    *,
    cost_waste_floor: int = DEFAULT_COST_WASTE_FLOOR,
    bands_version: str = DEFAULT_BANDS_VERSION,
) -> str:
    """The CI fixture variant: the identical script shape (the observer takes no scan cap —
    it reads exactly one run by id). Driven by
    ``tests/integration/test_wf_telemetry_observer_fixture.py``."""
    return build_observer_script(cost_waste_floor=cost_waste_floor, bands_version=bands_version)


# ─── the armed dead-man twin (observer_sweep) ─────────────────────────────────
#
# A SEPARATE scripted workflow (its crash mode must not entangle the observer), fired by a
# CronSource -> WorkflowAction. It asserts, on a DIFFERENT substrate: N terminal runs in the
# window => N observer-scans recorded. WHY required: a RunCompletionSource AUTO-DISABLES after
# MAX_CONSECUTIVE_FAILURES and CANNOT fire on its own non-firing — a dead observer is silent,
# and silence != health. The cron twin is the dead-man over the alarm.
#
# The list_runs read applies the no-silent-degrade invariant (#1323): a FULL page (returned
# count == requested limit) means more rows may exist unseen → the sweep CANNOT prove it saw
# all terminals → cannot-determine, NEVER a silent under-count read as "all scanned".

_SWEEP_BODY = r'''
import json


def _unwrap(input):
    """Accept a bare ``{watched_workflow_id, observer_workflow_id, after?}`` AND the cron
    WorkflowAction trigger envelope ``{"trigger": ..., "input": ...}`` (a cron fire carries
    no ``trigger.run``; the sweep takes its parameters from ``input``)."""
    if isinstance(input, dict) and "trigger" in input and "input" in input:
        return input.get("input") or {}
    return input or {}


async def _list_runs(args):
    """One list_runs read (full WfRun rows, account-scoped to this run's session)."""
    return await tool("list_runs", args)


def _page_runs(resp):
    """The runs list out of a list_runs tool return, or None if the read errored/degraded."""
    if not isinstance(resp, dict):
        return None
    if "error" in resp and "runs" not in resp:
        return None
    runs = resp.get("runs")
    if not isinstance(runs, list):
        return None
    return runs


async def main(input):
    phase("sweep")
    args = _unwrap(input)
    watched = args.get("watched_workflow_id")
    observer = args.get("observer_workflow_id")
    after = args.get("after")
    # The list_runs page limit the sweep requests. A returned page of exactly LIMIT rows is
    # the no-silent-degrade trip: more rows MAY exist unseen, so the count is unprovable.
    limit = args.get("limit", 200)

    if not isinstance(watched, str) or not isinstance(observer, str):
        log("sweep: missing watched/observer workflow id")
        return {
            "alarm": True,
            "verdict": "cannot-determine",
            "reason": "missing watched_workflow_id / observer_workflow_id",
            "bands_version": BANDS_VERSION,
        }

    # Count terminal runs of the watched workflow in the window. We read each terminal status
    # explicitly (the no-silent-omission default mirrors RunCompletionSource's all-terminal).
    n_runs = 0
    for status in ("completed", "errored", "cancelled"):
        req = {"workflow_id": watched, "status": status, "limit": limit, "account_wide": True}
        if after is not None:
            req["after"] = after
        resp = await _list_runs(req)
        runs = _page_runs(resp)
        if runs is None:
            log("sweep: list_runs read failed/degraded for", watched, status)
            return {
                "alarm": True,
                "verdict": "cannot-determine",
                "reason": "list_runs read failed for watched=%s status=%s" % (watched, status),
                "bands_version": BANDS_VERSION,
            }
        # No-silent-degrade (#1323): a FULL page means rows may exist beyond it that we did
        # NOT see — we cannot prove the terminal count, so we must NOT under-count silently.
        if len(runs) >= limit:
            log("sweep: truncated list_runs page (>= limit) for", watched, status)
            return {
                "alarm": True,
                "verdict": "cannot-determine",
                "reason": "truncated list_runs page for watched=%s status=%s (>= limit %d)"
                % (watched, status, limit),
                "bands_version": BANDS_VERSION,
            }
        n_runs += len(runs)

    # Count the observer's own scan-runs in the same window (its recorded scans).
    n_scans = 0
    req = {"workflow_id": observer, "status": "completed", "limit": limit, "account_wide": True}
    if after is not None:
        req["after"] = after
    resp = await _list_runs(req)
    scans = _page_runs(resp)
    if scans is None:
        log("sweep: list_runs read failed/degraded for observer", observer)
        return {
            "alarm": True,
            "verdict": "cannot-determine",
            "reason": "list_runs read failed for observer=%s" % observer,
            "bands_version": BANDS_VERSION,
        }
    if len(scans) >= limit:
        log("sweep: truncated list_runs page (>= limit) for observer", observer)
        return {
            "alarm": True,
            "verdict": "cannot-determine",
            "reason": "truncated list_runs page for observer=%s (>= limit %d)" % (observer, limit),
            "bands_version": BANDS_VERSION,
        }
    n_scans = len(scans)

    # The assertion: every terminal run must have produced an observer scan. A shortfall is
    # a dead/lagging observer — the dead-man rings.
    if n_runs > n_scans:
        log("sweep ALARM: n_runs", n_runs, ">", "n_scans", n_scans)
        return {
            "alarm": True,
            "verdict": "cannot-determine",
            "reason": "observer shortfall: %d terminal runs but only %d scans" % (n_runs, n_scans),
            "n_runs": n_runs,
            "n_scans": n_scans,
            "bands_version": BANDS_VERSION,
        }
    log("sweep ok: n_runs", n_runs, "n_scans", n_scans)
    return {
        "alarm": False,
        "verdict": "ok",
        "n_runs": n_runs,
        "n_scans": n_scans,
        "bands_version": BANDS_VERSION,
    }
'''


def build_sweep_script(*, bands_version: str = DEFAULT_BANDS_VERSION) -> str:
    """Return the production ``observer_sweep`` (dead-man twin) workflow source.

    A SEPARATE workflow from the observer (so the alarm's crash mode does not entangle the
    observed). It reads the terminal-run count via ``list_runs`` and the observer-scan count,
    and rings (``alarm: True`` / ``cannot-determine``) when terminal runs exceed scans OR when
    a ``list_runs`` page is truncated (the #1323 no-silent-degrade trip — a truncated page is
    NEVER read as "all scanned").
    """
    header = _py("BANDS_VERSION", bands_version)
    return header + "\n" + _SWEEP_BODY


def build_sweep_fixture_script(*, bands_version: str = DEFAULT_BANDS_VERSION) -> str:
    """CI fixture variant of the dead-man twin (identical shape). Driven by
    ``tests/integration/test_wf_telemetry_observer_fixture.py``."""
    return build_sweep_script(bands_version=bands_version)


# ─── deploy surface (the tool envelope a WorkflowCreate needs) ────────────────
#
# The observer reads host numbers via get_run; the sweep counts via list_runs. NEITHER
# declares http_request / bash / edit — the observer mutates nothing and touches no external
# artifact, and (structurally) CANNOT file a GitHub issue / obliging action-item (criterion
# 11: the issue-emission gating is enforced by the absence of http_request, not by a runtime
# check). Both workflows share this surface; the union is exactly {get_run, list_runs}.
REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="get_run"),
    ToolSpec(type="list_runs"),
]


def build_observer_workflow_create(
    *,
    name: str = "telemetry_observer",
    description: str | None = None,
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` for the production ``telemetry_observer``.

    Bundles the script (``build_observer_script``) with its tool surface
    (``REQUIRED_TOOLS = [get_run, list_runs]`` and NO http/bash/edit), so a deployer POSTs one
    object and the declared surface can never drift from the script that needs it (#1135).
    """
    return WorkflowCreate(
        name=name,
        description=description,
        script=build_observer_script(**script_kwargs),
        tools=list(REQUIRED_TOOLS),
    )


def build_sweep_workflow_create(
    *,
    name: str = "observer_sweep",
    description: str | None = None,
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` for the production ``observer_sweep`` dead-man
    twin. Shares the observer's ``REQUIRED_TOOLS`` (it reads via ``list_runs``; no mutation)."""
    return WorkflowCreate(
        name=name,
        description=description,
        script=build_sweep_script(**script_kwargs),
        tools=list(REQUIRED_TOOLS),
    )
