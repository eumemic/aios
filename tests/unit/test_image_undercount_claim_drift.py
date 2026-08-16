"""Guard: no source comment may claim the image under-count is absorbed by calibration.

``tokens.py`` states the invariant this guard enforces (issue #2050)::

    The scaling layer corrects a RATIO. It can only correct terms that are
    non-zero in the baseline. Any content this function does not count at
    all is invisible to calibration forever, because no coefficient
    multiplied by zero reaches a positive number.

LiteLLM's ``token_counter`` prices an ``image_url`` part at a near-constant
~89 tokens regardless of payload size (measured on litellm 1.97.0: 89 / 89 / 89
for 10 KB / 200 KB / 2 MB of image bytes, against 9,600 / 191,447 / 1,912,670
as text — 108x / 2,151x / 21,491x).  A constant against a linear truth is not a
ratio error, so **no** ``model_token_class_ratios`` coefficient can recover it.

Three copies of the falsified claim survived the first correction pass
(``context.py``), and the corrected docstring cross-references one of them —
so a reader following the pointer landed on a comment asserting the opposite.
This guard is what makes the correction hold across all copies: it fails on the
*coupling* (image under-count described as absorbed / bounded / harmless), not
on any one file's wording, so a re-introduction anywhere in ``src/`` trips it.

It must both REFUSE and PERMIT: ordinary calibration language about drifts that
ARE non-zero in the baseline (the tz / focal-channel rendering drifts) is
correct and must keep passing, as must ``tokens.py``'s own prose, which
discusses images and calibration together in order to deny the very claim.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "aios"

# Half-width of the SUBJECT window, in normalized characters.  Wide enough to
# span a multi-line comment block (the surviving events.py copy put "images
# undercount by ~55" and "absorbed by ... calibration" four lines apart, in two
# different sentences).
_WINDOW = 420

# Half-width of the REFUTATION window.  Deliberately much tighter than the
# subject window: a denial has to sit right next to the claim it denies, so
# that a re-introduced claim cannot be excused by an unrelated "cannot" several
# sentences away.  This is what lets the corrected comments state the false
# claim in order to REJECT it, without reopening the hole.
_REFUTATION_WINDOW = 200

# The absorption claim: calibration / the ratio layer soaks the error up.
_ABSORBED = re.compile(
    r"absorbed by\s+``?model_token_class_ratios``?"
    r"|absorbed by\s+``?model_token_ratio``?"
    r"|absorbed by (?:the )?calibration"
    r"|calibration absorbs",
    re.IGNORECASE,
)

# The minimization claim: the under-count is small / harmless / only marginal.
# Same falsehood in a different grammar, so pinning only ``absorbed by`` would
# let ``vision.py``'s wording back in.
_MINIMIZED = re.compile(
    r"small under-?count"
    r"|under-?count(?:ing)? only matters"
    r"|only matters near the window boundary",
    re.IGNORECASE,
)

# The subject the two claims above may not be made about.  ``vision/tz drift``
# is in here because the second surviving copy never names an image: it asserts
# absorption of "the documented vision/tz drifts below" BY CROSS-REFERENCE.
# Keying only on literal image words would let that copy through, which is
# exactly the class of miss that left three copies standing.
_IMAGE_SUBJECT = re.compile(
    r"image_url"
    r"|inlined image"
    r"|images under-?count"
    r"|per[- ]image"
    r"|vision session"
    r"|vision/tz drift",
    re.IGNORECASE,
)


# Explicit refutation. The corrected comments must be able to QUOTE the false
# claim in order to deny it -- that is how a reader learns the claim was tried
# and is wrong -- so quotation adjacent to one of these is permitted.  Bare
# proximity is not enough: the denial must be within
# :data:`_REFUTATION_WINDOW` of the claim itself.
_REFUTED = re.compile(
    r"\bcannot\b"
    r"|\bcan not\b"
    r"|\bnot\b in that class"
    r"|is NOT in that class"
    r"|\bwrongly\b"
    r"|\bfalse\b"
    r"|\bwas wrong\b"
    r"|earlier version"
    r"|previously claimed"
    r"|multiplied by zero",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Collapse whitespace and comment/docstring furniture to one flat line.

    A claim split across four ``#`` continuation lines must read as one
    sentence, or proximity matching silently misses it.
    """
    text = re.sub(r"^\s*#", " ", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text)


def stale_absorption_claims(text: str) -> list[str]:
    """Return excerpts asserting the image under-count is absorbed or small.

    A violation is a *coupling*: an absorption/minimization claim within
    :data:`_WINDOW` characters of image-under-count language.  Either half
    alone is legitimate prose.
    """
    flat = _normalize(text)
    hits: list[str] = []
    for pattern in (_ABSORBED, _MINIMIZED):
        for match in pattern.finditer(flat):
            lo = max(0, match.start() - _WINDOW)
            hi = min(len(flat), match.end() + _WINDOW)
            window = flat[lo:hi]
            if not _IMAGE_SUBJECT.search(window):
                continue
            rlo = max(0, match.start() - _REFUTATION_WINDOW)
            rhi = min(len(flat), match.end() + _REFUTATION_WINDOW)
            if _REFUTED.search(flat[rlo:rhi]):
                continue
            hits.append(window.strip())
    return hits


def test_no_source_comment_claims_the_image_undercount_is_absorbed() -> None:
    """REFUSAL over the real tree: every copy of the falsified claim is gone.

    Not just the one ``context.py`` corrected — the whole of ``src/aios``.
    """
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        for hit in stale_absorption_claims(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}: ...{hit}...")
    assert not offenders, (
        "A comment claims the image under-count is absorbed by calibration or is "
        "small. LiteLLM prices an image_url part at a near-constant ~89 tokens "
        "regardless of size, so the error is a CONSTANT against a linear truth: no "
        "ratio coefficient reaches it (see the tokens.py invariant, issue #2050).\n"
        + "\n".join(offenders)
    )


def test_guard_refuses_each_historical_sentence() -> None:
    """REFUSAL, pinned: the exact sentences that were in the tree, verbatim.

    Without this, a future edit could weaken the scanner to always-permit and
    the tree-wide test above would still pass, vacuously.
    """
    historical = {
        "context.py (corrected by aios#2141)": (
            "the append-time token counter in ``queries.append_event`` does not "
            "(and pays a small under-count per inlined image, absorbed by "
            "``model_token_class_ratios`` calibration)."
        ),
        "events.py:1436 (surviving copy 1)": (
            "        # NOTE(vision/tz): the USER ``delta`` was rendered without\n"
            "        # ``model``/``session_id`` and in the default UTC zone, so inlined\n"
            "        # images undercount by ~55 LiteLLM tokens each and a non-UTC account's\n"
            "        # envelope is a few tokens narrower than build time.  Both drifts are\n"
            "        # bounded and absorbed by ``model_token_class_ratios`` calibration in\n"
            "        # :func:`read_windowed_events` (see PR #218);"
        ),
        "events.py:1278 (surviving copy 2)": (
            "    an acceptable, bounded drift in the same class as\n"
            "    the documented vision/tz drifts below (absorbed by ``model_token_class_ratios``\n"
            "    calibration)."
        ),
        "vision.py:3 (surviving copy 3)": (
            "LiteLLM's :func:`token_counter` returns a flat ~85 tokens per\n"
            "``image_url`` part regardless of provider; under-counting only\n"
            "matters near the window boundary and provider rejection there is\n"
            "recoverable."
        ),
    }
    for label, sentence in historical.items():
        assert stale_absorption_claims(sentence), f"guard failed to refuse {label}"


def test_guard_permits_genuine_calibration_language() -> None:
    """PERMIT: a guard that only ever refuses is indistinguishable from one that
    refuses everything.  These must all pass.

    1. The tz / focal-channel rendering drifts ARE non-zero in the baseline, so
       calibration genuinely does absorb them — that sentence must survive.
    2. ``tokens.py`` discusses images, under-counting and calibration in one
       breath precisely to DENY absorption; flagging it would punish the
       correct text.
    3. The replacement wording this PR installs must itself pass, or the guard
       is unsatisfiable.
    """
    permitted = {
        "tz/focal drift, no image subject": (
            "        # NOTE(tz): a non-UTC account's envelope is a few tokens narrower\n"
            "        # than at build time.  That drift is bounded and absorbed by\n"
            "        # ``model_token_class_ratios`` calibration in\n"
            "        # :func:`read_windowed_events` (see PR #218)."
        ),
        "tokens.py denial prose": (
            "Any content this function does not count at all is invisible to\n"
            "calibration forever, because no coefficient multiplied by zero reaches\n"
            "a positive number.  Measured instance: message ``content`` parts of\n"
            '``type: "image_url"`` were not counted, a 7,461x under-count.\n'
            "Calibration cannot correct it."
        ),
        "the replacement wording": (
            "        # NOTE(vision): an inlined ``image_url`` part is priced by\n"
            "        # ``litellm.token_counter`` at a near-constant ~89 tokens regardless\n"
            "        # of payload size, so a data-URI image is under-counted without\n"
            "        # bound.  Calibration CANNOT recover it: the scaling layer corrects\n"
            "        # a ratio, and no coefficient multiplied by zero reaches a positive\n"
            "        # number (issue #2050; fix in aios#2073)."
        ),
    }
    for label, sentence in permitted.items():
        assert not stale_absorption_claims(sentence), (
            f"guard wrongly refused legitimate text ({label}): {stale_absorption_claims(sentence)}"
        )
