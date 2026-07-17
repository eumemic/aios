"""#1152 §6(w) — writer-coverage over the revoked/fulfilled kind taxonomy.

The cancel leaf (``harvest_session_cancel_markers``) classifies an already-closed
edge by its written ``error.kind``: ``kind ∈ REVOCATION_KINDS`` ⇒ REVOKED (counts as
revocation-context for §2.3 propagation), anything else — including unknown/absent —
⇒ FULFILLED (graceful; under-cancel is strictly safer and the lease ceiling
backstops). That default makes NEW kinds silently graceful, so every kind an ``Err``
writer can actually produce must be a *conscious* classification: this test
enumerates the writer call sites by AST scan and fails when a kind appears that is
in neither ``REVOCATION_KINDS`` nor the fulfilled ledger below — the anti-drift
guard the spec demands (§1: "a test asserting every Err-writing call site's kind is
classified").
"""

from __future__ import annotations

import ast
from pathlib import Path

import aios
from aios.services.sessions import REVOCATION_KINDS

SRC_ROOT = Path(aios.__file__).parent

# Kinds an Err writer can produce that classify FULFILLED (graceful): written by the
# servicer itself — its own ``error`` tool, its harness-erroring path (the model
# failed past its retry budget and the session's own step latches + answers), or the
# run's own script/host failure answering its caller through its terminal record.
# A REVOCATION is a closure by someone OTHER than the servicer (stop/timeout/caller
# terminal/caller archived) — keep this ledger in lockstep with spec §1.
FULFILLED_KINDS: frozenset[str] = frozenset(
    {
        # the servicer quiesced owing its answer (the totality backstop)
        "no_return",
        # the run answered but its output failed the request's acceptance contract
        "output_schema_violation",
        # session harness-erroring: the servicer's own step latches its death
        "model_terminal_error",
        "model_call_deadline",
        "model_refusal",
        "model_workflow_run_errored",
        "model_workflow_invalid_shape",
        "child_errored",
        "spend_cap_exceeded",
        "provider_auth_conflict",
        "model_provider_not_configured",
        # the session errored its OWN turn: its context didn't fit and it
        # exhausted the shrink-on-retry ladder (#1792) — servicer's own erroring
        "context_overflow",
        # run-side script/host failures (HostErrorKind + step dispatch arms): the
        # run answers its caller via its own terminal record
        "author_exception",
        "too_wide_fanout",
        "script_host_crash",
        "script_host_timeout",
        "script_host_spawn_failed",
        "too_many_agents",
        "bad_tool_call",
        "not_implemented",
    }
)


def _kind_from_dict(node: ast.expr) -> str | None:
    """The ``"kind"`` value of a dict literal, when both key and value are constants."""
    if not isinstance(node, ast.Dict):
        return None
    for key, value in zip(node.keys, node.values, strict=True):
        if (
            isinstance(key, ast.Constant)
            and key.value == "kind"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    return None


def _kind_value_constants(node: ast.expr) -> list[str]:
    """String constants in KIND-VALUE position: a bare literal, or the literal
    operand(s) of an ``x or "<lit>"`` fallback (the ``terminal.get("kind") or
    "author_exception"`` shape). Deliberately does NOT walk into calls/subscripts —
    a ``.get("kind")`` lookup is a dict KEY, not a kind value (the read-side
    ``error_kind=(data.get("error") or {}).get("kind")`` shape must not count)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BoolOp):
        return [
            v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
    return []


def _collect_writer_kinds() -> dict[str, list[str]]:
    """kind → call sites, from the three origination shapes an Err writer takes:

    * an ``error={"kind": <lit>}`` kwarg (``Err(...)``, ``fail_open_child_requests_conn``,
      ``fail_all_open_requests`` and friends);
    * an ``error_kind=<lit>`` kwarg (``_complete_run`` / ``_latch_errored_turn`` /
      ``HostOutcome`` origination sites), including the ``x or "<lit>"`` fallback shape;
    * an ``error_kind = "<lit>"`` assignment (the schema-violation arm).
    """
    kinds: dict[str, list[str]] = {}

    def record(kind: str, path: Path, lineno: int) -> None:
        kinds.setdefault(kind, []).append(f"{path.relative_to(SRC_ROOT)}:{lineno}")

    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "error" and (kind := _kind_from_dict(kw.value)) is not None:
                        record(kind, path, node.lineno)
                    elif kw.arg == "error_kind":
                        for lit in _kind_value_constants(kw.value):
                            record(lit, path, node.lineno)
            elif (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "error_kind" for t in node.targets)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                record(node.value.value, path, node.lineno)
    return kinds


def test_scanner_sees_the_known_writers() -> None:
    """Non-vacuity: the scan must surface the kinds we KNOW the tree writes — if a
    refactor changes the writer shapes, this fails before the coverage assert goes
    silently green on an empty set."""
    kinds = set(_collect_writer_kinds())
    assert {
        "cancelled",
        "timeout",
        "child_gone",
        "engine_semantics_changed",
        "nondeterministic_replay",
        "no_return",
    } <= kinds, f"writer scan lost known kinds; saw only {sorted(kinds)}"


def test_every_err_writer_kind_is_classified() -> None:
    kinds = _collect_writer_kinds()
    unclassified = {
        kind: sites
        for kind, sites in kinds.items()
        if kind not in REVOCATION_KINDS and kind not in FULFILLED_KINDS
    }
    assert not unclassified, (
        "Err-writer kinds with no revoked/fulfilled classification (they would "
        "silently default to FULFILLED at the cancel leaf). Categorize each in "
        "aios.services.sessions.REVOCATION_KINDS (a withdrawal by someone other "
        "than the servicer) or in this test's FULFILLED_KINDS ledger (the "
        f"servicer's own answer/erroring): {unclassified}"
    )


def test_ledgers_are_disjoint() -> None:
    overlap = REVOCATION_KINDS & FULFILLED_KINDS
    assert not overlap, f"a kind cannot be both revoked and fulfilled: {sorted(overlap)}"
