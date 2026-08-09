from pathlib import Path

from scripts.partition_integration_files import build_shards


def test_partition_covers_every_integration_file_exactly_once() -> None:
    root = Path("tests/integration")
    expected = sorted(root.glob("test_*.py"))
    shards = build_shards(root, 8)
    assigned = [path for shard in shards for path in shard]

    assert sorted(assigned) == expected
    assert len(assigned) == len(set(assigned))
    assert all(shards)


def test_partition_is_deterministic() -> None:
    root = Path("tests/integration")
    assert build_shards(root, 8) == build_shards(root, 8)
