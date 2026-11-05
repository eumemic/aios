"""``pipeline-provenance`` required check: a pipeline PR merges into master only with provenance.

Root cause this closes (aios#2572): a seat direct-merged an unstamped ``pipeline:v2`` PR with the
``eumemic`` token, bypassing the dev-pipeline reconciler's ``_do_merge``. Branch protection runs
``enforce_admins=true``, so once ``pipeline-provenance`` is a required status check on master the
same merge returns HTTP 405 whatever token issues it.

A PR is a PIPELINE PR when its head ref starts with ``dev-pipeline/`` OR it carries ``pipeline:v2``.
A pipeline PR targeting master passes only when BOTH hold:

  (a) it is reviewed-for-the-LIVE-head, by the reconciler's own rule (``owner()`` branch 8):
      the legacy ``reviewed:green@<sha12>`` label for exactly this head, OR the flat
      ``reviewed:green`` label AND the canonical review record in the PR's TRUSTED comment thread
      showing a green verdict for exactly this head (company#349). A record for an older head, a
      veto at this head, an untrusted author or an unreadable thread all FAIL CLOSED;
  (b) exactly one ``risk:tier-N`` label, N an integer in ``1..AUTO_MERGE_MAX_TIER``.

Every other PR (and any PR whose base is not master) passes immediately -- the check always
reports a conclusion, so as a required check it never strands a PR in "expected".

The review-record grammar below is VENDORED VERBATIM from eumemic-company
``app/src/eumemic_company_app/workflows/dev_pipeline_reconciler.py`` (origin/master
b0fa9e3e3636577bc2a1573e1406c3f7f06ac12b): ``_strip_fenced_blocks``, ``_review_verdict_token``,
``_advisory_ci_bot_body``, ``_canonical_review_comment``, ``_review_green_at_head``; and the
trusted-author predicate from ``workflows/comment_trust.py`` (``_comment_trusted``). Only type
annotations were added. Re-vendor (do not re-derive) when the reconciler's grammar moves: a
grammar that diverges from the reconciler's would green PRs the reconciler would never merge, or
block PRs it legitimately merged.

Stdlib only; the workflow runs it from the BASE branch under ``pull_request_target`` so a PR can
never edit the gate that judges it.

Usage::

    python scripts/pipeline_provenance.py --pr pr.json --comments comments.jsonl

``pr.json`` is the live ``GET /repos/{o}/{r}/pulls/{n}``; ``comments.jsonl`` is every
``GET /repos/{o}/{r}/issues/{n}/comments`` item, one JSON object per line (``gh api --paginate
--jq '.[]'``). Exit 0 = pass, 1 = provenance missing, 2 = unreadable input (also a failure).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_BRANCH = "master"
PIPELINE_BRANCH_PREFIX = "dev-pipeline/"
LABEL_PIPELINE_V2 = "pipeline:v2"
LABEL_REVIEWED = "reviewed:green"
LABEL_REVIEWED_PREFIX = "reviewed:green@"
LABEL_RISK_PREFIX = "risk:tier-"
#: reconciler ``REVIEWED_SHA_LEN``: the legacy ``reviewed:green@<sha>`` stamp's sha-prefix length.
REVIEWED_SHA_LEN = 12

#: The highest risk tier the pipeline may merge without a human. THE single knob here.
#: Source of truth: eumemic-company ``app/infra/lanes/aios-substrate.json`` ->
#: ``"auto_merge_max_tier": 4`` (company#453 lane IaC), which ``reregister_workflows.py`` renders
#: into the live reconciler as ``AUTO_MERGE_MAX_TIER = 4`` (``app/infra/lanes/aios-substrate.lock.json``).
#: The module default ``AUTO_MERGE_MAX_TIER = 2`` in ``dev_pipeline_reconciler.py`` (and the
#: "=3" in its ~L6573 comment) are NOT what the aios lane runs. Keep this in lockstep with the lane.
AUTO_MERGE_MAX_TIER = 4

#: eumemic-company ``comment_trust.TRUSTED_COMMENT_LOGINS`` / ``TRUSTED_AUTHOR_ASSOCIATIONS``.
TRUSTED_COMMENT_LOGINS: tuple[str, ...] = ("eumemic", "eumemic-bot[bot]")
TRUSTED_AUTHOR_ASSOCIATIONS: tuple[str, ...] = ("OWNER", "MEMBER")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNREADABLE = 2


# ─── vendored: comment_trust._comment_trusted ────────────────────────────────


def _comment_trusted(c: Any) -> bool:
    """True iff the comment's author is ours: an allowlisted ``user.login`` OR an
    ``author_association`` of OWNER/MEMBER. Missing author fields -> False (FAIL CLOSED)."""
    if not isinstance(c, dict):
        return False
    user = c.get("user")
    login = user.get("login") if isinstance(user, dict) else None
    if isinstance(login, str) and login in TRUSTED_COMMENT_LOGINS:
        return True
    assoc = c.get("author_association")
    return isinstance(assoc, str) and assoc.upper() in TRUSTED_AUTHOR_ASSOCIATIONS


# ─── vendored: dev_pipeline_reconciler review-record grammar ─────────────────


def _strip_fenced_blocks(text: str | None) -> str:
    """Remove fenced code blocks per CommonMark's actual closing rules.

    company#357/#360. My first version used a regex with a backreference, which
    review broke twice with real forgeries:

      * a 4-backtick opener "closed" by a 3-backtick line -> the rest of the
        block, INCLUDING A VERDICT, leaked into the visible text as authority
      * a closing line carrying an info string (```python) accepted as a close

    CommonMark: a closing fence must use the SAME character, be at least as long
    as the opener, be indented at most 3 spaces, and carry NOTHING but the
    delimiter and trailing whitespace. An unterminated fence runs to end of
    document. Getting this wrong forges a green verdict, so it is implemented
    explicitly rather than as a clever pattern.
    """
    out: list[str] = []
    fence_char = None
    fence_len = 0
    fence_col = 0
    for line in (text or "").splitlines():
        opening = re.match(r"^([ \t]*)(`{3,}|~{3,})(.*)$", line)
        if opening:
            # CommonMark's "up to three spaces" is a column rule, not a
            # character-count rule. Tabs advance to the next four-column tab
            # stop, so e.g. ` \t``` reaches column four and is indented code,
            # never a fence delimiter. Applying this to both openers and
            # closers keeps every authority consumer on the same scanner.
            # company#404: an indented (4+ column) fence is, at top level, an
            # indented code block -- but the reviewer nests fenced evidence in
            # list items (`- **Failing test:**\n    ```bash ...`), where CommonMark
            # reads it as a fence at the item's content column. Left unstripped
            # its backticks paired into a multi-line code SPAN and the whole
            # record was rejected as malformed (company#401, 09:50Z). Treating an
            # indented fence as a fence only removes MORE text from the authority
            # scan (indented lines never supply authority), so it cannot forge a
            # verdict; it only stops evidence from vetoing its own record. The
            # column rule still applies to CLOSERS so a shorter/indented closer
            # cannot end a longer block early (the #357/#360 forgeries).
            column = 0
            for char in opening.group(1):
                column = column + 1 if char == " " else column + (4 - column % 4)
            # A closer may sit at most three columns deeper than its opener
            # (CommonMark, relative to the container); deeper is content.
            if fence_char is not None and column > fence_col + 3:
                opening = None
        if fence_char is None:
            if opening:
                fence_char = opening.group(2)[0]
                fence_len = len(opening.group(2))
                fence_col = column
                continue
            out.append(line)
            continue
        # inside a fence: only a conforming closer ends it
        if (
            opening
            and opening.group(2)[0] == fence_char
            and len(opening.group(2)) >= fence_len
            and not opening.group(3).strip()
        ):
            fence_char = None
            fence_len = 0
        continue
    return "\n".join(out)


# The advisory CI reviewer's GitHub App login (aios#2404 / aios#2483). Its comments are findings
# for a human, never a verdict; see _canonical_review_comment for the exact release shape.
CI_BOT_REVIEW_LOGIN = "eumemic-bot[bot]"


def _review_verdict_token(body: str) -> bool:
    """True when the RAW body carries ANY verdict-shaped text (aios#2483, PR #444 review).

    Deliberately broad: it only ever DENIES the advisory-bot release below, so a false positive
    costs master's fail-closed parse, never a skipped veto. Measured: 0 hits on the 55
    post-aios#2404 eumemic-bot[bot] comments.
    """
    return bool(
        re.search(
            r"(?i)\bverdict\b[\s*_`]*[:=\u2014\u2013-]|[*_]{2}[\s`]*verdict\b|\bverdict\b[\s:]*[*_]{2}"
            r"|(?i:(?m:^[ \t]{0,3}#{1,6}[ \t]*verdict\b))"
            r"|\bverdict\b\W{0,5}(?:pass|fail|block|green|red|approve|reject|needs|merge|lgtm)"
            r"|\b(?:MERGE-SAFE|NEEDS-CHANGES|BLOCK-SPEC-WRONG|BLOCK-BUILD-VIOLATES-SPEC)\b"
            r"|\bREQUEST(?:ED)?[ _-]CHANGES\b|\bCHANGES[ _-]REQUESTED\b|\bNEEDS?[ _-]+CHANGES?\b"
            r"|#{1,6}\s+Re-verify|reviewed:green@"
            r"|[*_]{2}[\s`]*(?:PASS(?:ED|ES)?|FAIL(?:ED|S|URE)?|BLOCK(?:S|ED|ING|ER)?|GREEN|RED|CLEAN"
            r"|APPROVED?|LGTM|REJECT(?:ED)?)\b"
            r"|(?m:^\W{0,6}(?:result|status|outcome|decision|conclusion)\W{0,6}[ \t]*[:=\u2014\u2013-]"
            r"[ \t]*\W{0,6}(?:pass|fail|block|green|red|approve|reject|needs|merge|lgtm))"
            r"|\b(?:do\s+not|don'?t|must\s+not|cannot|can'?t)\s+merge\b|\bnot\s+merge[- ]?(?:safe|ready)\b"
            # The verdict-LINE SHAPE with any word (`**MAYBE** at `<sha>``).
            r"|[*_]{2}[^*_\r\n]{1,80}[*_]{2}[ \t]*(?:at|@)[ \t]*`?[0-9a-fA-F]{7,40}\b",
            body,
        )
    ) or bool(
        # Case-SENSITIVE bare veto words (lower-case `fail` is ordinary prose in findings).
        re.search(
            r"\b(?:FAIL(?:ED|S|URE)?|BLOCK(?:ED|S|ING|ER)?|REJECT(?:ED)?|DO NOT MERGE)\b", body
        )
    )


def _advisory_ci_bot_body(body: Any) -> bool:
    """The POSITIVE shape of an advisory CI-bot comment: ends with exactly one publisher marker
    ``<!-- eumemic-bot-review:<40-hex> -->`` and carries no verdict-shaped text anywhere."""
    if not isinstance(body, str) or body.count("eumemic-bot-review:") != 1:
        return False
    if not re.search(r"<!-- eumemic-bot-review:[0-9a-f]{40} -->[ \t\r\n]*\Z", body):
        return False
    return not _review_verdict_token(body)


def _canonical_review_comment(
    body: Any, candidate_head: str | None, author: str | None = None
) -> tuple[str, str | None] | None:
    """Parse one durable PR-comment review record outside quoted/code Markdown.

    Returns ``None`` for unrelated comments, otherwise ``(verdict, provenance)`` where verdict is
    ``green``, ``veto``, or ``malformed`` and provenance is a full SHA, ``stale``, or ``None``.
    Only a full 40-hex SHA can establish exact-head evidence; short prefixes never do.  Blockquotes,
    fenced code, and indented code are excluded before parsing: review comments routinely quote an
    earlier record, and quoted/fenced text is discussion or data, never an active authority signal.
    """
    if not isinstance(body, str):
        return None
    # Reject the whole record when its authority signal is embedded in Markdown whose rendering is
    # context-sensitive.  This deliberately strict preflight is smaller and safer than attempting a
    # partial CommonMark implementation (lazy/nested quotes and fence-length rules are easy to get
    # subtly wrong).  The sole permitted code span is a full-SHA provenance span.
    # THE ADVISORY CI BOT IS DISCUSSION -- AND NOTHING ELSE IS RELEASED (aios#2483).
    #
    # The eumemic-bot[bot] CI reviewer (a TRUSTED author) opens every comment with `### Code
    # review`, writes findings with markdown links and `<!-- inspected ... -->` HTML comments, ends
    # with its publisher marker `<!-- eumemic-bot-review:<sha> -->`, and -- since aios#2404, by
    # design ("advisory channel, not a gate") -- carries NO verdict. The heading raised
    # canonical_hint, the HTML comments tripped inert_markdown, and every one of its comments
    # parsed ('malformed', None): 97/97 on the last 40 aios PRs. The verify gate fails closed on
    # ANY trusted malformed comment, so it could never pass on a PR the bot had touched
    # (aios#2482), whatever the reconciler's own canonical record (company#383) said.
    #
    # AN ALLOWLIST, NOT A DENYLIST (PR #444 review). Round 1 released every `### Code review`
    # without a verdict token; the token was a denylist of veto spellings, and review found five
    # it missed (`## Verdict` + `NEEDS CHANGES`, `Verdict FAIL at`, `**Needs changes**`,
    # `Result: FAIL`, ...) -- a trusted veto turned into a skip, and next to a canonical green the
    # gate PASSED where master failed closed. So the release is now a POSITIVE shape, and every
    # other comment parses exactly as on master. Released to None only when ALL hold:
    #   1. the caller vouches the AUTHOR is the CI bot login (`author=`); a body alone never
    #      qualifies, so body-only readers (fix-park, memoised review) keep master's behaviour;
    #   2. the body ENDS with the bot's publisher marker `<!-- eumemic-bot-review:<40-hex> -->`
    #      and carries it exactly once;
    #   3. the RAW body (fences, quotes, HTML comments, link destinations included) has no
    #      verdict-shaped text at all -- belt and braces: the bot is verdict-free by contract, so
    #      if it ever writes one, that comment falls back to master's fail-closed parse.
    # A new veto spelling therefore cannot reopen this for any author but the bot, and the bot's
    # comment was never an authority channel for the gate (it is advisory; the record of record
    # is the reconciler's canonical record). And the release can never hide a verdict the grammar
    # below would have read: the required marker IS an HTML comment, which trips inert_markdown,
    # so the same body can only ever parse ('malformed', None) or None without this line.
    if author == CI_BOT_REVIEW_LOGIN and _advisory_ci_bot_body(body):
        return None
    canonical_hint = bool(
        re.search(r"(?i)(?:### Code review|#{1,6}\s+Re-verify|reviewed:green@)", body)
    )
    # FENCED AUTHORITY IS A FORGERY ATTEMPT, NOT AN ABSENCE (company#357).
    #
    # Allowing fences (so real evidence-bearing reviews parse at all) must not
    # downgrade a fenced-only verdict from "malformed" to None. Callers treat
    # those differently and it matters: None means SKIP THIS COMMENT, malformed
    # means FAIL THE GATE CLOSED. A record whose only authority signal sits
    # inside a fence is someone pasting a verdict, so it must still fail closed
    # rather than being quietly ignored.
    fenced_only_authority = False
    if canonical_hint:
        stripped = _strip_fenced_blocks(body)
        fenced_only_authority = not re.search(
            r"(?i)(?:### Code review|#{1,6}\s+Re-verify|reviewed:green@)", stripped
        )
    # CommonMark code spans pair backtick runs of equal length. Track each delimiter length
    # independently so a closing tick from one ordinary span cannot become the opener for a later
    # span (a whole-body regex makes that mistake across prose such as the #2045 review).
    # SCAN FOR SPANS OUTSIDE FENCES (company#357). A fence delimiter is itself a
    # run of backticks, so scanning the raw body counted ```bash ... ``` as an
    # unterminated multi-line code SPAN and disqualified every real record —
    # the same fence-vs-evidence conflation as the predicate below, one layer
    # down. Remove fenced blocks first; what remains is prose, where a genuine
    # multi-line span really is ambiguous.
    span_source = _strip_fenced_blocks(body)
    open_code_spans: dict[int, int] = {}
    multiline_code_span = False
    for run in re.finditer(r"`+", span_source):
        length = len(run.group(0))
        if length in open_code_spans:
            start = open_code_spans.pop(length)
            if "\n" in span_source[start : run.start()]:
                multiline_code_span = True
                break
        else:
            open_code_spans[length] = run.end()

    # Inert Markdown must never become review authority merely because a whole-body regex can see it.
    # Deliberately reject canonical-looking records containing HTML comments/raw HTML or inline-link
    # destinations rather than attempting to reproduce GitHub's full HTML/CommonMark renderer. The
    # visible link label is discussion; its hidden destination is never an authority channel.
    inert_markdown = (
        bool(re.search(r"<!--[\s\S]*?-->", body))
        or bool(re.search(r"<\s*/?\s*[A-Za-z][^>\r\n]*>", body))
        or bool(re.search(r"\]\s*\(", body))
    )
    # A FENCE IS EVIDENCE, NOT AMBIGUITY (company#357).
    #
    # This predicate disqualified any record containing a fenced or indented
    # block. Measured against 40 real review records from live PRs: ALL 40
    # contain a code fence, because the pipeline REQUIRES "machine-replayable
    # commands/fixtures for every finding" and that evidence is posted fenced.
    # So the format the pipeline demands was the format its own parser refused,
    # and `_verify_gate_exact_head_ready` returned canonical-review-malformed for
    # every genuine review — including all 9 real MERGE-SAFE records. A gate that
    # cannot accept any real evidence is a gate nothing can pass.
    #
    # The property that actually matters is unchanged: AUTHORITY MUST NOT COME
    # FROM INSIDE A FENCE. The stripping pass below already enforces exactly
    # that by parsing only the visible remainder. Presence of a fence is
    # therefore not evidence of ambiguity, and rejecting on it conflated
    # "contains quoted material" with "its verdict is quoted".
    #
    # Blockquotes stay disqualifying at the preflight: a quoted record is a
    # RESTATEMENT of someone else's verdict, which is a different risk from
    # attaching evidence to your own.
    ambiguous_markdown = (
        inert_markdown
        or bool(re.search(r"(?m)^\s{0,3}>", body))
        # INDENTED code is treated like a fence (company#404): the reviewer DOES
        # attach evidence this way (five prose records on company#401 in one
        # morning, each `    locks=sorted(...)`), and rejecting the whole record
        # for it fails the gate closed past a green canonical record. The
        # property that matters is unchanged -- AUTHORITY MUST NOT COME FROM AN
        # INDENTED LINE -- and the visible-remainder pass below already drops
        # indented lines before reading verdicts, so `    reviewed:green@<sha>`
        # still cannot become authority; that case is asserted explicitly here.
        or bool(
            re.search(
                r"(?mi)^(?: {4}|\t)[^\r\n]*(?:reviewed:green@|Verdict:|Re-verify|### Code review)",
                body,
            )
        )
        # Prose may contain code spans (the real #2045 review does). Reject authority in one.
        or bool(re.search(r"`[^`\n]*(?:reviewed:green@|Verdict:|Re-verify)[^`\n]*`", body, re.I))
        # CommonMark code spans may cross line boundaries; authority inside one is inert.
        or multiline_code_span
    )
    if fenced_only_authority:
        return ("malformed", None)
    if ambiguous_markdown:
        return ("malformed", None) if canonical_hint else None
    # ONE SHARED FENCE SCANNER (review round 5). This loop tracked only the fence
    # CHARACTER, so it ended a ````-opened block at the first ``` line and
    # accepted a closer carrying an info string. Both leak the remainder of the
    # block — INCLUDING A VERDICT — into the visible text, which review
    # demonstrated as two working forgeries of a green verdict:
    #     ````text / evidence / ``` / **MERGE-SAFE** at <head>. / ````
    #     ```text  / evidence / ```python / **MERGE-SAFE** at <head>. / ```
    # Three places were scanning fences with three different sets of rules; they
    # now share _strip_fenced_blocks, which implements the CommonMark closing
    # rules (same char, at least as long, <=3 spaces indent, nothing trailing).
    visible: list[str] = []
    for line in _strip_fenced_blocks(body).splitlines():
        if re.match(r"^\s*>", line) or re.match(r"^(?: {4}|\t)", line):
            continue
        visible.append(line)
    body = "\n".join(visible)
    code_review = bool(re.search(r"(?mi)^### Code review\s*$", body))
    reverify = re.search(r"(?mi)^#{1,6}\s+Re-verify\s+@([0-9a-f]{7,40})\b([^\r\n]*)$", body)
    green_stamps = re.findall(r"(?i)\breviewed:green@([0-9a-f]{7,40})\b", body)
    canonical = code_review or reverify is not None or bool(green_stamps)
    if not canonical:
        return None

    # THE PRODUCER'S ACTUAL GRAMMAR (company#357 corpus, 40 live records).
    #
    # This parser only recognised `**Verdict: <x>.**`. The configured reviewer
    # emits `**NEEDS-CHANGES** at \`<sha>\`.` — so ALL 40 sampled real records
    # parsed as ('malformed', None), which means `_verify_gate_exact_head_ready`
    # returns canonical-review-malformed for every genuine review, green ones
    # included. A gate that cannot accept any real evidence is a gate nothing can
    # pass; that is a live defect, not a memoisation detail.
    #
    # Both spellings are accepted now. The closed vocabulary is taken from the
    # corpus (NEEDS-CHANGES 30, BLOCK-BUILD-VIOLATES-SPEC 1, MERGE-SAFE 9); an
    # unrecognised marker still yields "malformed" rather than being assumed.
    markers = re.findall(r"(?mi)^\*\*Verdict:\s*([^*\r\n]+?)\.?\*\*", body)
    markers += re.findall(
        r"(?mi)^\*\*(?:Verdict:\s*)?(MERGE-SAFE|NEEDS-CHANGES|BLOCK-SPEC-WRONG|BLOCK-BUILD-VIOLATES-SPEC)\*\*"
        r"(?=\s*(?:at\b|$|[.,]))",
        body,
    )
    # THE LIVE PRODUCER'S THIRD SPELLING (company#383). Measured 2026-09-02 on
    # eumemic/aios: 64 of 66 `### Code review` records since 08-25 were `malformed`
    # under this parser, 45 of them because the reviewer writes `Verdict: **fail**`
    # (plain prefix, bold VALUE) — neither regex above matches, verdicts stays empty,
    # and every fail-closed consumer (verify gate, #380 lift, fixround driver) then
    # correctly declined. aios#2331 had no machine transition for 68 hours on exactly
    # this. The company#357 corpus fix chased one spelling; the producer moved to
    # another days later. Accept the plain-prefix line, keep the closed vocabulary.
    markers += re.findall(r"(?mi)^Verdict:\s*\*\*([^*\r\n]+?)\.?\*\*", body)
    verdicts: list[str] = []
    for marker in markers:
        raw = marker.strip().rstrip(".").strip().upper()
        if raw in ("PASS", "APPROVED", "MERGE-SAFE"):
            verdicts.append("green")
        elif raw == "FAIL":
            # The live producer's vocabulary is pass/fail (its agent schema says so:
            # {verdict:'pass'|'fail'}). `fail` is a veto; `pass` is already green above.
            verdicts.append("veto")
        elif (
            raw.startswith("REQUEST CHANGES")
            or raw.startswith("NEEDS-CHANGES")
            or raw.startswith("BLOCK")
        ):
            # `BLOCK` bare, `BLOCK-SPEC-WRONG`, `BLOCK-BUILD-VIOLATES-SPEC` -- the live
            # reviewer emits all three (company#404: bare `BLOCK` at #401 09:50Z).
            verdicts.append("veto")
        else:
            return ("malformed", None)
    if reverify is not None:
        heading = reverify.group(2).upper()
        if "NEEDS-CHANGES" in heading or "REQUEST CHANGES" in heading:
            verdicts.append("veto")
        elif re.search(r"\b(?:PASS|APPROVED)\b", heading):
            verdicts.append("green")
    if green_stamps:
        verdicts.append("green")
    if not verdicts or len(set(verdicts)) != 1:
        return ("malformed", None)
    verdict = verdicts[-1]

    heads = re.findall(
        r"(?im)(?:\b(?:head|head commit|review of head)\s*[:=` ]\s*|re-verify\s+@)"
        r"([0-9a-f]{7,40})\b",
        body,
    )
    # `**NEEDS-CHANGES** at \`<sha>\`.` — the producer's provenance syntax.
    heads += re.findall(
        r"(?im)^\*\*(?:Verdict:\s*)?"
        r"(?:MERGE-SAFE|NEEDS-CHANGES|BLOCK-SPEC-WRONG|BLOCK-BUILD-VIOLATES-SPEC)\.?\*\*\s+at\s+"
        r"`?(?<![0-9a-fA-F])([0-9a-f]{40})(?![0-9a-fA-F])`?",
        body,
    )
    # `**Verdict: PASS** at `<sha>`.` / `Verdict: **fail** at `<sha>`` — the live producer's
    # verdict-line carrier with its own pass/fail vocabulary (company#383/#384). The
    # verdict word was already validated above; here only the SHA is read, and only from
    # the verdict LINE (horizontal whitespace only — `\s` would let `at` on a later line
    # bind), so it is an explicit carrier, not prose.
    heads += re.findall(
        r"(?im)^(?:\*\*Verdict:\s*[^*\r\n]+?\.?\*\*|Verdict:\s*\*\*[^*\r\n]+?\.?\*\*)[ \t]+at[ \t]+"
        r"`?(?<![0-9a-fA-F])([0-9a-f]{40})(?![0-9a-fA-F])`?",
        body,
    )
    heads.extend(green_stamps)
    heads = list(dict.fromkeys(value.lower() for value in heads))
    # PROSE PROVENANCE, FAIL-CLOSED (company#383). The live producer puts the reviewed
    # SHA in prose ("remain satisfied at `<sha>`", "At `<sha>`, all properties hold",
    # "**Verdict: PASS** at `<sha>`") rather than on a `head:` line. None of the
    # carriers above see it, so 19 records with a parseable verdict still died here.
    # Accept a SHA from the VISIBLE prose only when it is UNAMBIGUOUS: exactly one
    # distinct full 40-hex value in the whole visible body (fences and quotes were
    # stripped above). Two distinct SHAs, a short prefix, or none -> still malformed.
    # This never widens an existing carrier: a `head:` / `at` / stamp carrier that
    # already produced a SHA is used as before; the prose scan runs only when none did.
    # VETO ONLY (company#384 adversarial review, BLOCKER): a sole prose SHA does not
    # prove it is the REVIEWED head — a `pass` that mentions only the merge-base would
    # become exact-head GREEN evidence for the merge-base. A wrongly-bound VETO is
    # fail-safe (it holds; a head mismatch reads as stale and nothing lifts on it); a
    # wrongly-bound GREEN is merge evidence. So prose provenance can only ever bind a
    # veto. Green records must carry an explicit carrier (`at `<sha>``, `head:`,
    # `reviewed:green@`, `Re-verify @`) — which the reconciler's own canonical record
    # does.
    if not heads and verdict == "veto":
        prose = list(
            dict.fromkeys(
                m.lower()
                for m in re.findall(r"(?<![0-9a-fA-F])([0-9a-f]{40})(?![0-9a-fA-F])", body)
            )
        )
        if len(prose) == 1:
            heads = prose
    # A short prefix is not exact provenance, even if it happens to prefix today's live head.
    if len(heads) != 1 or len(heads[0]) != 40:
        return ("malformed", None)
    provenance = heads[0]
    candidate = (candidate_head or "").lower()
    if provenance != candidate:
        return (verdict, "stale")
    return (verdict, candidate)


def _review_green_at_head(comments: Any, head_sha: Any) -> bool | None:
    """company#349: does the canonical review record show a GREEN verdict for exactly ``head_sha``?

    ``True``  -- at least one trusted, well-formed exact-head green record and NO exact-head veto
                 (latest record per author wins; authors compose conservatively, as the verify gate).
    ``False`` -- the thread was read and no such green exists (incl. a green for an OLDER head: the
                 commit-after-review mutation) or an exact-head veto is active.
    ``None``  -- the thread is unreadable. ``owner()`` treats anything but ``True`` as not-reviewed,
                 so an unreadable thread can never launder into "reviewed".
    Malformed records neither grant nor revoke here: a grant needs a well-formed exact-head green.
    """
    if not isinstance(comments, list):
        return None
    if not (isinstance(head_sha, str) and head_sha):
        return False
    head = head_sha.lower()
    latest = {}
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        user = comment.get("user") or {}
        login = user.get("login") if isinstance(user, dict) else None
        parsed = _canonical_review_comment(comment.get("body"), head, author=login)
        if not parsed:
            continue
        verdict, provenance = parsed
        if verdict not in ("green", "veto") or provenance != head:
            continue
        latest[login or ""] = verdict
    if any(v == "veto" for v in latest.values()):
        return False
    return any(v == "green" for v in latest.values())


# ─── the gate ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: tuple[str, ...]


def is_pipeline_pr(head_ref: str, labels: frozenset[str]) -> bool:
    """Pipeline PR = head ref under ``dev-pipeline/`` OR the ``pipeline:v2`` label."""
    return head_ref.startswith(PIPELINE_BRANCH_PREFIX) or LABEL_PIPELINE_V2 in labels


def risk_tier_problem(labels: frozenset[str], cap: int = AUTO_MERGE_MAX_TIER) -> str | None:
    """None when exactly one ``risk:tier-N`` with integer N in ``1..cap`` is present."""
    tiers = sorted(n for n in labels if n.startswith(LABEL_RISK_PREFIX))
    if not tiers:
        return "no risk:tier-N label"
    if len(tiers) != 1:
        return "risk tier must be single-valued, found {}".format(", ".join(tiers))
    raw = tiers[0][len(LABEL_RISK_PREFIX) :]
    if not re.fullmatch(r"[0-9]+", raw):
        return f"{tiers[0]} is not an integer tier"
    tier = int(raw)
    if not 1 <= tier <= cap:
        return f"{tiers[0]} is outside the auto-merge range 1..{cap} (AUTO_MERGE_MAX_TIER)"
    return None


def review_problem(labels: frozenset[str], head_sha: str, comments: list[Any]) -> str | None:
    """None when reviewed-for-head by the reconciler's ``owner()`` rule (legacy sha-scoped label
    for THIS head, or flat label + trusted green record for THIS head)."""
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        return f"live head sha {head_sha!r} is not a full sha"
    if LABEL_REVIEWED_PREFIX + head_sha[:REVIEWED_SHA_LEN] in labels:
        return None
    if LABEL_REVIEWED not in labels:
        legacy = sorted(n for n in labels if n.startswith(LABEL_REVIEWED_PREFIX))
        if legacy:
            return "{} is stale for head {}".format(", ".join(legacy), head_sha)
        return "no reviewed:green label"
    trusted = [c for c in comments if _comment_trusted(c)]
    if _review_green_at_head(trusted, head_sha) is not True:
        return (
            "reviewed:green present but no trusted canonical review record shows a green "
            f"verdict for head {head_sha} (stale, vetoed, untrusted or missing)"
        )
    return None


def evaluate(pr: dict[str, Any], comments: list[Any]) -> Verdict:
    """The whole decision over the live PR payload and its full comment thread."""
    base = (pr.get("base") or {}).get("ref")
    head = pr.get("head") or {}
    head_ref = head.get("ref") or ""
    head_sha = (head.get("sha") or "").lower()
    labels = frozenset(
        lab["name"] for lab in pr.get("labels") or [] if isinstance(lab, dict) and "name" in lab
    )
    if base != BASE_BRANCH:
        return Verdict(True, (f"base {base!r} is not {BASE_BRANCH}: not gated",))
    if not is_pipeline_pr(head_ref, labels):
        return Verdict(True, ("not a pipeline PR: not gated",))
    reasons = [
        r for r in (review_problem(labels, head_sha, comments), risk_tier_problem(labels)) if r
    ]
    if reasons:
        return Verdict(False, tuple(reasons))
    return Verdict(True, (f"pipeline PR reviewed green at {head_sha} with an in-cap risk tier",))


def _load(pr_path: Path, comments_path: Path) -> tuple[dict[str, Any], list[Any]]:
    pr = json.loads(pr_path.read_text())
    if not isinstance(pr, dict):
        raise ValueError("PR payload is not a JSON object")
    comments = [json.loads(line) for line in comments_path.read_text().splitlines() if line.strip()]
    return pr, comments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=Path, required=True)
    parser.add_argument("--comments", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        pr, comments = _load(args.pr, args.comments)
    except (OSError, ValueError) as exc:
        print(f"pipeline-provenance: FAIL (unreadable input, fail closed): {exc}")
        return EXIT_UNREADABLE
    verdict = evaluate(pr, comments)
    print("pipeline-provenance: %s" % ("PASS" if verdict.passed else "FAIL"))
    for reason in verdict.reasons:
        print(f"  - {reason}")
    return EXIT_PASS if verdict.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
