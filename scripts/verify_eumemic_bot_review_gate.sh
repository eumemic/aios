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
#   3. the live attack: a fake agent that forges published=true and emits no
#      evidence, driven through the REAL launcher, must leave the safety net
#      firing;
#   4. the permit half: an honest agent must still publish AND still produce
#      published=true, which is what proves the strip costs nothing.
#
# Usage: scripts/verify_eumemic_bot_review_gate.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FAIL=0
step() { printf '\n=== %s ===\n' "$1"; }

step "1. unit suite"
# These tests import nothing from the `aios` package, but tests/unit/conftest.py
# does. In a bare checkout without the project installed, run them in isolation
# rather than reporting a collection error as a test failure.
if python3 -c "import aios" 2>/dev/null; then
  python3 -m pytest tests/unit/test_eumemic_bot_review.py -q -p no:cacheprovider || FAIL=1
else
  echo "note: \`aios\` is not importable here; running these tests without the package conftest."
  ISO="$(mktemp -d)"
  mkdir -p "$ISO/tests/unit"
  ln -s "$ROOT/scripts" "$ISO/scripts"
  ln -s "$ROOT/.github" "$ISO/.github"
  cp tests/unit/test_eumemic_bot_review.py "$ISO/tests/unit/"
  (cd "$ISO" && python3 -m pytest tests/unit/test_eumemic_bot_review.py -q -p no:cacheprovider) || FAIL=1
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
  out="$(cd "$1" && python3 -m pytest tests/unit/test_eumemic_bot_review.py \
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
  mkdir -p "$1/scripts" "$1/tests/unit" "$1/.github/workflows"
  cp scripts/eumemic_bot_review.py "$1/scripts/"
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

step "2b. self-check: a collection error must NOT be counted as a kill"
D="$TMP/broken-collect"; mk_tree "$D"
printf '\nimport nonexistent_module_xyz\n' >> "$D/tests/unit/test_eumemic_bot_review.py"
SELF="$(FAIL=0; assert_killed "$D" "test_agent_cannot_reach_the_actions_control_files" "self-check" 2>&1)"
if grep -q INCONCLUSIVE <<<"$SELF"; then
  echo "OK — a collection error reports INCONCLUSIVE, not 'MUTANT KILLED'"
else
  echo "BROKEN — the kill-check still fails open on a collection error: $SELF"; FAIL=1
fi

step "3+4. forged-\$GITHUB_OUTPUT attack, then the honest agent, through the REAL launcher"
ATK="$(mktemp -d)"; trap 'rm -rf "$TMP" "$ATK"' EXIT
mkdir -p "$ATK/bin"
# Attacker: forge the net's own input, emit NO evidence.
cat >"$ATK/bin/codex" <<'EOF'
#!/usr/bin/env bash
[ -n "${GITHUB_OUTPUT:-}" ] && { echo "published=true" >> "$GITHUB_OUTPUT"; echo "AGENT-SAW-GITHUB_OUTPUT" >&2; }
# Round 3: the net's OUTPUT channel, not just its input. The "did not post"
# warning is written to $GITHUB_STEP_SUMMARY, so a writable handle there lets
# the agent forge a "review posted" banner over the operator's own account of
# what happened. $GITHUB_STATE survives into the job's post-steps.
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && { echo "### eumemic-bot review posted :white_check_mark:" >> "$GITHUB_STEP_SUMMARY"; echo "AGENT-SAW-GITHUB_STEP_SUMMARY" >&2; }
[ -n "${GITHUB_STATE:-}" ] && { echo "forged=1" >> "$GITHUB_STATE"; echo "AGENT-SAW-GITHUB_STATE" >&2; }
out=""; while [ $# -gt 0 ]; do [ "$1" = "--output-last-message" ] && out="$2"; shift; done
cat >/dev/null
printf '### Code review\n\nNo actionable findings. LGTM.\n' > "$out"
EOF
# Honest agent: actually runs the mandated commands.
cat >"$ATK/bin/codex-honest" <<'EOF'
#!/usr/bin/env bash
out=""; while [ $# -gt 0 ]; do [ "$1" = "--output-last-message" ] && out="$2"; shift; done
cat >/dev/null
L=$(git --no-pager diff "$BASE_SHA...$HEAD_SHA" | wc -l)
S=$(git --no-pager diff "$BASE_SHA...$HEAD_SHA" | sha256sum | cut -d' ' -f1)
printf '### Code review\n\nBLOCKING: a real finding.\n\n<!-- inspected: lines=%s sha256=%s -->\n' "$L" "$S" > "$out"
EOF
chmod +x "$ATK/bin/codex" "$ATK/bin/codex-honest"
cat >"$ATK/server.py" <<'EOF'
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        open(sys.argv[2], "a").write("POSTED\n")
        self.send_response(201); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"html_url": "https://example/c/1", "body": b["body"]}).encode())
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF
PORT=8759
POSTED="$ATK/posted.log"; : >"$POSTED"
python3 "$ATK/server.py" "$PORT" "$POSTED" & SRV=$!
sleep 1
sed "s#https://api.github.com#http://127.0.0.1:$PORT#" scripts/eumemic_bot_review.py >"$ATK/launcher.py"
H_SHA="$(git rev-parse HEAD)"; B_SHA="$(git rev-parse HEAD~1)"
run_case() {  # $1 = agent binary name
  GHO="$ATK/gho.$1"; : >"$GHO"; : >"$POSTED"
  SUMMARY="$ATK/summary.$1"; : >"$SUMMARY"
  cp "$ATK/bin/$1" "$ATK/bin/codex_run"; mv "$ATK/bin/codex_run" "$ATK/bin/codex"
  PATH="$ATK/bin:$PATH" GITHUB_OUTPUT="$GHO" GITHUB_ENV="$ATK/env" GITHUB_PATH="$ATK/path" \
  GITHUB_STEP_SUMMARY="$SUMMARY" GITHUB_STATE="$ATK/state" \
  GH_TOKEN=tok OAI_PROXY_API_KEY=key REPO=eumemic/aios PR_NUMBER=0 \
  HEAD_SHA="$H_SHA" BASE_SHA="$B_SHA" REVIEW_MODEL=gpt-5.6-sol \
  REVIEW_SANDBOX_MODE=danger-full-access \
  python3 "$ATK/launcher.py" >"$ATK/out.$1" 2>"$ATK/err.$1"
  echo $?
}
RC="$(run_case codex)"
echo "attack: launcher exit=$RC posted=$([ -s "$POSTED" ] && echo YES || echo NO) GITHUB_OUTPUT='$(tr -d '\n' <"$ATK/gho.codex")'"
if grep -q AGENT-SAW-GITHUB_OUTPUT "$ATK/err.codex" || grep -q published=true "$ATK/gho.codex"; then
  echo "  VULNERABLE (input) — the agent forged the safety net's input"; FAIL=1
else
  echo "  SAFE (input) — agent blind to GITHUB_OUTPUT, net still fires"
fi
# The net's OUTPUT channel: the warning a human actually reads.
if grep -q AGENT-SAW-GITHUB_STEP_SUMMARY "$ATK/err.codex" || \
   grep -q "review posted" "$ATK/summary.codex"; then
  echo "  VULNERABLE (output) — the agent forged the 'did not post' warning's own channel"; FAIL=1
else
  echo "  SAFE (output) — agent blind to GITHUB_STEP_SUMMARY, the warning cannot be overwritten"
fi
if grep -q AGENT-SAW-GITHUB_STATE "$ATK/err.codex" || [ -s "$ATK/state" ]; then
  echo "  VULNERABLE (state) — the agent reached GITHUB_STATE, which survives into post-steps"; FAIL=1
else
  echo "  SAFE (state) — agent blind to GITHUB_STATE"
fi
RC="$(run_case codex-honest)"
echo "honest: launcher exit=$RC posted=$([ -s "$POSTED" ] && echo YES || echo NO) GITHUB_OUTPUT='$(tr -d '\n' <"$ATK/gho.codex-honest")'"
if [ "$RC" = 0 ] && [ -s "$POSTED" ] && grep -q published=true "$ATK/gho.codex-honest"; then
  echo "  PERMITS — a genuine review still publishes and still signals"
else
  echo "  BROKEN — the strip suppressed a legitimate review"; FAIL=1
fi
kill $SRV 2>/dev/null

if [ "$FAIL" = 0 ]; then echo; echo "ALL CHECKS PASSED"; else echo; echo "CHECKS FAILED"; fi
exit "$FAIL"
