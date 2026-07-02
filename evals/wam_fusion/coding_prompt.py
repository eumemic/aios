"""Coding-tier prompt construction + candidate-source extraction.

A coding item asks the model to FIX source file(s). We show it the task + the full
current content of each source file (at base), and require it to return the COMPLETE
corrected content of each file in a fenced block tagged with the file path, so the
extraction is unambiguous and the scorer can drop each file in verbatim.
"""

from __future__ import annotations

import re

CODING_SYSTEM = (
    "You are an expert Python engineer fixing a bug in the aios codebase. You are given a "
    "task and the CURRENT FULL CONTENT of one or more source files. Produce the CORRECTED "
    "FULL CONTENT of each file you change. A held-out test suite (which you cannot see) will "
    "judge your fix. Return EACH changed file as a fenced code block whose info string is the "
    "EXACT file path, e.g.:\n\n"
    "```src/aios/models/attenuation.py\n<the complete corrected file content>\n```\n\n"
    "Return the WHOLE file, not a diff and not an ellipsis. Only include files you change. Do "
    "not add commentary inside the code blocks."
)


def build_prompt(
    task: str, base_sources: dict[str, str], expected_paths: list[str] | None = None
) -> str:
    """Build the generation prompt.

    ``base_sources`` may be EMPTY for a new-file-creation task (the golden PR
    created the source file, so the pre-PR snapshot legitimately has nothing to
    show) — ``expected_paths`` then names the file(s) to create, mirroring how an
    existing file's path is shown alongside its content. Passing neither is a
    caller bug (was a StopIteration crash on `next(iter(base_sources))`)."""
    paths = list(base_sources)
    for p in expected_paths or []:
        if p not in paths:
            paths.append(p)
    if not paths:
        raise ValueError("build_prompt: no base sources and no expected paths")
    parts = [f"TASK:\n{task}\n"]
    if base_sources:
        parts.append("CURRENT SOURCE FILE(S):")
        for path, content in base_sources.items():
            parts.append(f"\n----- FILE: {path} -----\n```\n{content}\n```")
    to_create = [p for p in paths if p not in base_sources]
    if to_create:
        parts.append(
            "FILE(S) TO CREATE (they do not exist yet — write the complete new file):\n"
            + "\n".join(f"- {p}" for p in to_create)
        )
    parts.append(
        "\nReturn the complete corrected content of each file you change, each in a fenced "
        "block whose info string is the exact file path (e.g. ```" + paths[0] + ")."
    )
    return "\n".join(parts)


# Match a fenced block whose info string is a path under src/ (the file we asked for).
_FENCE = re.compile(
    r"```[ \t]*(?P<path>(?:src/|tests/)?[\w./-]+\.py)[ \t]*\r?\n(?P<body>.*?)\r?\n```",
    re.DOTALL,
)


def extract_sources(text: str | None, expected_paths: list[str]) -> dict[str, str]:
    """Pull each expected file's corrected content from the model's fenced blocks.

    Matches blocks tagged with the exact path; falls back to a single untagged block
    when exactly one file was expected (a lenient path for models that omit the tag).
    """
    if not text:
        return {}
    out: dict[str, str] = {}
    for m in _FENCE.finditer(text):
        p = m.group("path").strip()
        if p in expected_paths and p not in out:
            out[p] = m.group("body")
    # Lenient single-file fallback: one expected file, one untagged ``` block.
    if not out and len(expected_paths) == 1:
        m = re.search(r"```(?:python)?\r?\n(.*?)\r?\n```", text, re.DOTALL)
        if m:
            out[expected_paths[0]] = m.group(1)
    return out
