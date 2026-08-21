#!/usr/bin/env python3
"""Fail when a PR silently removes files that still exist on its current base."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def find_suspicious_deletions(
    files: list[dict[str, Any]], description: str, base_paths: set[str]
) -> list[str]:
    """Return undocumented, unreplaced removals reported by GitHub's PR API."""
    description = description.casefold()
    replacement_blobs = {
        file.get("sha")
        for file in files
        if file.get("status") != "removed" and file.get("sha") is not None
    }
    return sorted(
        filename
        for file in files
        if file.get("status") == "removed"
        and (filename := str(file["filename"])) in base_paths
        and filename.casefold() not in description
        and file.get("sha") not in replacement_blobs
    )


def failure_message(paths: list[str]) -> str:
    noun = "file" if len(paths) == 1 else "files"
    listing = "\n".join(f"  {path}" for path in paths)
    return (
        f"This PR deletes {len(paths)} {noun} that exist on master and are not mentioned "
        f"in its description:\n{listing}\n"
        "This usually means the branch is STALE (forked before those files landed).\n"
        "Rebase onto master and re-check. If the deletion IS intended, say so in the PR body."
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_pr_deletions.py PR_FILES_JSON BASE_PATHS")
    files = json.loads(Path(sys.argv[1]).read_text())
    if not isinstance(files, list):
        raise TypeError("PR files response must be a JSON list")
    base_paths = set(Path(sys.argv[2]).read_text().splitlines())
    description = f"{os.environ.get('PR_TITLE', '')}\n{os.environ.get('PR_BODY', '')}"
    suspicious = find_suspicious_deletions(files, description, base_paths)
    if not suspicious:
        print("No undocumented deletions of files on the current base.")
        return 0
    print(failure_message(suspicious), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
