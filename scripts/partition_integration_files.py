#!/usr/bin/env python3
"""Deterministically balance complete integration-test files across CI runners."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_shards(root: Path, count: int) -> list[list[Path]]:
    files = sorted(root.glob("test_*.py"))
    if not files:
        raise RuntimeError(f"no integration tests found under {root}")
    shards: list[list[Path]] = [[] for _ in range(count)]
    weights = [0] * count
    for path in sorted(files, key=lambda item: (-weight(item), str(item))):
        shard = min(range(count), key=lambda index: (weights[index], index))
        shards[shard].append(path)
        weights[shard] += weight(path)
    assigned = [path for shard in shards for path in shard]
    if sorted(assigned) != files or len(assigned) != len(set(assigned)):
        raise RuntimeError("integration partition must cover every file exactly once")
    return shards


def weight(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    # Test count approximates ordinary DB work. Files that boot their own
    # Postgres pay a substantial fixed Docker/migration startup premium.
    tests = sum(
        line.lstrip().startswith(("def test_", "async def test_")) for line in source.splitlines()
    )
    return max(tests, 1) + (15 if "PostgresContainer(" in source else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("tests/integration"))
    args = parser.parse_args()
    if args.count < 1 or not 0 <= args.shard < args.count:
        parser.error("shard must satisfy 0 <= shard < count")
    selected = build_shards(args.root, args.count)[args.shard]
    print(*(str(path) for path in selected), sep="\n")


if __name__ == "__main__":
    main()
