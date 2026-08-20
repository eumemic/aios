#!/usr/bin/env python3
"""Emit a GitHub-PR-files-shaped comparison of the current base and PR trees."""

from __future__ import annotations

import json
import subprocess
import sys


def tree(commit: str) -> dict[str, str]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", commit], text=True
    )
    entries: dict[str, str] = {}
    for record in output.split("\0"):
        if not record:
            continue
        metadata, filename = record.split("\t", 1)
        _mode, _kind, sha = metadata.split()
        entries[filename] = sha
    return entries


def compare(base: str, head: str) -> list[dict[str, str]]:
    base_tree = tree(base)
    head_tree = tree(head)
    files: list[dict[str, str]] = []
    for filename in sorted(base_tree.keys() | head_tree.keys()):
        if filename not in head_tree:
            files.append({"filename": filename, "status": "removed", "sha": base_tree[filename]})
        elif filename not in base_tree:
            files.append({"filename": filename, "status": "added", "sha": head_tree[filename]})
        elif base_tree[filename] != head_tree[filename]:
            files.append({"filename": filename, "status": "modified", "sha": head_tree[filename]})
    return files


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: pr_files_from_trees.py BASE_SHA HEAD_SHA")
    json.dump(compare(sys.argv[1], sys.argv[2]), sys.stdout)


if __name__ == "__main__":
    main()
