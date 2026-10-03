#!/usr/bin/env bash
# Re-verification in one command, for the review round that follows this one.
#
# The reviewer asked that re-verifying this fix be "a suite re-run plus one
# mutant confirming the stripped vars, NOT another full round trip". This is
# that. It runs:
#
#   1. the unit suite;
#   2. THE mutant: put GITHUB_OUTPUT back in the agent env and confirm a test
#      DIES (a guard nobody has seen fail has not been shown to guard anything);
#   2a. mutants for the second miss-report channel (#2424): the `::` defusing
#      of harness output, and the net's own `::error::` annotation;
#   3. the live attack: an EMPTY-environment agent forges and floods the run
#      summary via its constant path; the net's annotation must still fire.
#
# Usage: scripts/verify_eumemic_bot_review_gate.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FAIL=0
# The project venv has pytest + pyyaml; a bare system python3 usually does not,
# which made every mutant below report WRONG FAILURE (collection died).
PY="python3"; [ -x "$ROOT/.venv/bin/python" ] && PY="$ROOT/.venv/bin/python"
step() { printf '\n=== %s ===\n' "$1"; }

step "1. unit suite"
# These tests import nothing from the `aios` package, but tests/unit/conftest.py
# does. In a bare checkout without the project installed, run them in isolation
# rather than reporting a collection error as a test failure.
if python3 -c "import aios" 2>/dev/null; then
  "$PY" -m pytest tests/unit/test_eumemic_bot_review.py -q -p no:cacheprovider || FAIL=1
else
  echo "note: \`aios\` is not importable here; running these tests without the package conftest."
  ISO="$(mktemp -d)"
  mkdir -p "$ISO/tests/unit"
  ln -s "$ROOT/scripts" "$ISO/scripts"
  ln -s "$ROOT/.github" "$ISO/.github"
  ln -s "$ROOT/docs" "$ISO/docs"
  cp tests/unit/test_eumemic_bot_review.py "$ISO/tests/unit/"
  (cd "$ISO" && "$PY" -m pytest tests/unit/test_eumemic_bot_review.py -q -p no:cacheprovider) || FAIL=1
  rm -rf "$ISO"
fi

# A kill must be ATTRIBUTED, not inferred from pytest being unhappy.
#
# The previous version treated ANY nonzero pytest exit as "MUTANT KILLED". A
# reviewer hit that for real: with pyyaml missing, collection died and this
# script printed "MUTANT KILLED — the strip is genuinely guarded" having run
# ZERO tests. That is the same fail-open shape this whole PR exists to remove,
# sitting inside the tool that certifies the fix. A kill now requires the
# NAMED test to be reported failed, and a collection error is INCONCLUSIVE
# (which fails the run) rather than a pass.
assert_killed() {  # $1 = tree, $2 = expected failing test, $3 = label
  local out rc
  out="$(cd "$1" && "$PY" -m pytest tests/unit/test_eumemic_bot_review.py \
        -q -p no:cacheprovider 2>&1)"; rc=$?
  if [ "$rc" = 0 ]; then
    echo "MUTANT SURVIVED ($3) — the suite is green with the fix reverted"; FAIL=1; return
  fi
  if grep -qiE 'error(s)? during collection|INTERNALERROR|ModuleNotFoundError' <<<"$out"; then
    echo "INCONCLUSIVE ($3) — pytest never COLLECTED, so nothing was verified:"
    grep -iE 'ModuleNotFoundError|error' <<<"$out" | sed 's/^/    /' | head -3
    FAIL=1; return
  fi
  if grep -q "FAILED tests/unit/test_eumemic_bot_review.py::$2" <<<"$out"; then
    echo "MUTANT KILLED ($3) — $2 failed, as required"
  else
    echo "WRONG FAILURE ($3) — expected $2 to fail; it failed for another reason:"
    grep -E '^(FAILED|ERROR)' <<<"$out" | sed 's/^/    /' | head -5
    FAIL=1
  fi
}

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mk_tree() {  # $1 = dest
  mkdir -p "$1/scripts" "$1/tests/unit" "$1/.github/workflows" "$1/docs"
  cp scripts/eumemic_bot_review.py "$1/scripts/"
  cp docs/eumemic-bot-review.md "$1/docs/"
  cp tests/unit/test_eumemic_bot_review.py "$1/tests/unit/"
  cp .github/workflows/eumemic-bot-review.yml "$1/.github/workflows/"
}

step "2. mutants: each control var removed from the strip (must KILL a named test)"
# All five, not just the one we were attacked through: the round-2 strip covered
# three of five, and no mutant existed for the missing two, so nothing caught it.
for VAR in GITHUB_OUTPUT GITHUB_ENV GITHUB_PATH GITHUB_STEP_SUMMARY GITHUB_STATE; do
  D="$TMP/strip-$VAR"; mk_tree "$D"
  python3 - "$D/scripts/eumemic_bot_review.py" "$VAR" <<'PY'
import sys
p, var = sys.argv[1], sys.argv[2]
text = open(p).read()
needle = f'    "{var}",\n'
assert needle in text, f"{var} is not in _STRIPPED_ENV at all — the fix is gone"
open(p, "w").write(text.replace(needle, "", 1))
PY
  assert_killed "$D" "test_control_variables_are_stripped_by_name_not_only_by_path" \
    "$VAR removed from the name list"
done

# The value-based strip is a separate guard; break it independently.
D="$TMP/no-path-filter"; mk_tree "$D"
python3 - "$D/scripts/eumemic_bot_review.py" <<'PY'
import sys
p = sys.argv[1]
text = open(p).read()
needle = " and _CONTROL_PATH_MARKER not in v.lower()"
assert needle in text, "the control-path value filter is gone"
open(p, "w").write(text.replace(needle, "", 1))
PY
assert_killed "$D" "test_agent_cannot_reach_the_actions_control_files" \
  "path-value filter removed"

# The coupling guard that licenses the `==`/startswith equivalent mutant.
D="$TMP/wide-regex"; mk_tree "$D"
python3 - "$D/scripts/eumemic_bot_review.py" <<'PY'
import sys
p = sys.argv[1]
text = open(p).read()
assert "{64}" in text, "the evidence regex no longer pins 64 hex"
open(p, "w").write(text.replace("{64}", "{1,64}", 1))
PY
assert_killed "$D" "test_evidence_regex_will_not_even_match_a_short_digest" \
  "evidence regex widened to admit a short digest"

step "2a. mutants: the miss-report's second channel (must KILL a named test)"
# Issue #2424: env-stripping cannot hide a constant path, so the miss is also
# reported as an ::error:: annotation from the net's own stdout. Both halves of
# that need a mutant: the defusing that stops the agent forging it, and the
# annotation itself.
D="$TMP/no-defuse"; mk_tree "$D"
python3 - "$D/scripts/eumemic_bot_review.py" <<'PY'
import sys
p = sys.argv[1]
text = open(p).read()
needle = "        _emit(defuse_workflow_commands(text), stream)"
assert needle in text, "harness output is no longer defused"
open(p, "w").write(text.replace(needle, "        _emit(text, stream)", 1))
PY
assert_killed "$D" "test_harness_output_cannot_forge_or_stop_the_annotation_channel" \
  "harness output echoed without :: defusing"

D="$TMP/no-annotation"; mk_tree "$D"
python3 - "$D/.github/workflows/eumemic-bot-review.yml" <<'PY'
import sys
p = sys.argv[1]
lines = open(p).read().splitlines(keepends=True)
kept = [l for l in lines if 'echo "::error title=eumemic-bot review did not post' not in l]
assert len(kept) == len(lines) - 1, "the net's ::error:: annotation is gone"
open(p, "w").write("".join(kept))
PY
assert_killed "$D" "test_a_flooded_or_unwritable_summary_cannot_suppress_the_annotation" \
  "net's ::error:: annotation removed"

step "2b. self-check: a collection error must NOT be counted as a kill"
D="$TMP/broken-collect"; mk_tree "$D"
printf '\nimport nonexistent_module_xyz\n' >> "$D/tests/unit/test_eumemic_bot_review.py"
SELF="$(FAIL=0; assert_killed "$D" "test_agent_cannot_reach_the_actions_control_files" "self-check" 2>&1)"
if grep -q INCONCLUSIVE <<<"$SELF"; then
  echo "OK — a collection error reports INCONCLUSIVE, not 'MUTANT KILLED'"
else
  echo "BROKEN — the kill-check still fails open on a collection error: $SELF"; FAIL=1
fi

step "3+4. the net's two channels, attacked through the REAL workflow step"
# The previous live attack drove the single-phase launcher (#2404), which no
# longer exists: the agent phase now needs sudo + setpriv, and publication moved
# to a separate job. What #2424 adds is that a miss is reported on a channel the
# agent cannot write to, so attack THAT: an agent with an EMPTY environment that
# globs the constant control-file directory, forges a banner into the summary
# and floods it past 1 MiB, followed by the net step itself.
ATK="$(mktemp -d)"; trap 'rm -rf "$TMP" "$ATK"' EXIT
CTRL="$ATK/_temp/_runner_file_commands"; mkdir -p "$CTRL"
SUMMARY="$CTRL/step_summary_net"; : >"$SUMMARY"
env -i /bin/bash -c "for f in $CTRL/step_summary_*; do
  echo '### eumemic-bot review posted :white_check_mark:' >> \"\$f\"
  head -c 1258291 /dev/zero | tr '\\0' x >> \"\$f\"
done"
NET="$("$PY" -c 'import sys, yaml
steps = yaml.safe_load(open(sys.argv[1]))["jobs"]["publish"]["steps"]
print(next(s for s in steps if "GITHUB_STEP_SUMMARY" in s.get("run", ""))["run"])' \
  .github/workflows/eumemic-bot-review.yml)"
OUT="$(GITHUB_STEP_SUMMARY="$SUMMARY" bash -e -c "$NET")"
SIZE="$(wc -c <"$SUMMARY")"
echo "summary after attack: ${SIZE} bytes (runner skips upload above 1048576)"
if grep -q '^::error title=eumemic-bot review did not post::' <<<"$OUT"; then
  echo "  SAFE — the ::error:: annotation fires from the net's own stdout"
else
  echo "  VULNERABLE — the flooded/forged summary was the only report"; FAIL=1
fi

if [ "$FAIL" = 0 ]; then echo; echo "ALL CHECKS PASSED"; else echo; echo "CHECKS FAILED"; fi
exit "$FAIL"
