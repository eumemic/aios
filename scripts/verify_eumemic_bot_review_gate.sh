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

step "2. mutant: GITHUB_OUTPUT restored to the agent env (must KILL a test)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/tests/unit" "$TMP/.github/workflows"
cp scripts/eumemic_bot_review.py "$TMP/scripts/"
cp tests/unit/test_eumemic_bot_review.py "$TMP/tests/unit/"
cp .github/workflows/eumemic-bot-review.yml "$TMP/.github/workflows/"
python3 - "$TMP/scripts/eumemic_bot_review.py" <<'PY'
import sys
p = sys.argv[1]
text = open(p).read()
needle = '    "GITHUB_OUTPUT",\n'
assert needle in text, "GITHUB_OUTPUT is not stripped at all — the fix is gone"
open(p, "w").write(text.replace(needle, "", 1))
PY
if (cd "$TMP" && python3 -m pytest tests/unit/test_eumemic_bot_review.py -q -p no:cacheprovider >/dev/null 2>&1); then
  echo "MUTANT SURVIVED — no test distinguishes the stripped env from the unstripped one"; FAIL=1
else
  echo "MUTANT KILLED — the strip is genuinely guarded"
fi

step "3+4. forged-\$GITHUB_OUTPUT attack, then the honest agent, through the REAL launcher"
ATK="$(mktemp -d)"; trap 'rm -rf "$TMP" "$ATK"' EXIT
mkdir -p "$ATK/bin"
# Attacker: forge the net's own input, emit NO evidence.
cat >"$ATK/bin/codex" <<'EOF'
#!/usr/bin/env bash
[ -n "${GITHUB_OUTPUT:-}" ] && { echo "published=true" >> "$GITHUB_OUTPUT"; echo "AGENT-SAW-GITHUB_OUTPUT" >&2; }
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
  cp "$ATK/bin/$1" "$ATK/bin/codex_run"; mv "$ATK/bin/codex_run" "$ATK/bin/codex"
  PATH="$ATK/bin:$PATH" GITHUB_OUTPUT="$GHO" GITHUB_ENV="$ATK/env" GITHUB_PATH="$ATK/path" \
  GH_TOKEN=tok OAI_PROXY_API_KEY=key REPO=eumemic/aios PR_NUMBER=0 \
  HEAD_SHA="$H_SHA" BASE_SHA="$B_SHA" REVIEW_MODEL=gpt-5.6-sol \
  REVIEW_SANDBOX_MODE=danger-full-access \
  python3 "$ATK/launcher.py" >"$ATK/out.$1" 2>"$ATK/err.$1"
  echo $?
}
RC="$(run_case codex)"
echo "attack: launcher exit=$RC posted=$([ -s "$POSTED" ] && echo YES || echo NO) GITHUB_OUTPUT='$(tr -d '\n' <"$ATK/gho.codex")'"
if grep -q AGENT-SAW-GITHUB_OUTPUT "$ATK/err.codex" || grep -q published=true "$ATK/gho.codex"; then
  echo "  VULNERABLE — the agent forged the safety net's input"; FAIL=1
else
  echo "  SAFE — agent blind to GITHUB_OUTPUT, net still fires"
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
