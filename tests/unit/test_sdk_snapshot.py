"""Drift check: ``packages/aios-sdk/aios_sdk/_generated/`` must match what
``scripts/regen-client.sh`` would produce from the current ``openapi.json``.

Sibling to :mod:`tests.unit.test_openapi_snapshot` which catches the
upstream step (FastAPI app → openapi.json). This catches the
downstream step (openapi.json → generated SDK). Without this guard a
contributor could update routes, regen ``openapi.json`` to satisfy
:mod:`test_openapi_snapshot`, commit both, and forget to regen the
SDK — leaving downstream consumers with stale types until someone
notices.

The test invokes ``openapi-python-client`` directly (the same command
``scripts/regen-client.sh`` uses) into a tmpdir, then walks both trees
in parallel. Difference → fail with the remediation command.

Filed as #369.
"""

from __future__ import annotations

import filecmp
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATED_DIR = _REPO_ROOT / "packages" / "aios-sdk" / "aios_sdk" / "_generated"
_OPENAPI_SPEC = _REPO_ROOT / "openapi.json"
# Caches and similar build artifacts the generator may emit alongside
# the source files. Mirrors the ``--exclude`` set in
# :file:`scripts/regen-client.sh`.
_IGNORE_NAMES: frozenset[str] = frozenset({".ruff_cache", "__pycache__"})


def _walk_relative_files(root: Path) -> set[Path]:
    """All files under ``root``, returned as paths relative to ``root``.

    Skips directory components present in :data:`_IGNORE_NAMES` and any
    file inside such a directory.
    """
    out: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _IGNORE_NAMES for part in rel.parts):
            continue
        out.add(rel)
    return out


def test_sdk_generated_matches_committed(tmp_path: Path) -> None:
    """The committed ``_generated/`` tree must match a fresh regen.

    If this test fails, the remediation is ``scripts/regen-client.sh``
    followed by ``git add packages/aios-sdk/aios_sdk/_generated``.
    """
    assert _GENERATED_DIR.is_dir(), f"committed _generated/ missing at {_GENERATED_DIR}"
    assert _OPENAPI_SPEC.is_file(), f"openapi.json missing at {_OPENAPI_SPEC}"

    work = tmp_path / "regen"
    work.mkdir()

    # Invokes the same generator command as ``scripts/regen-client.sh``.
    # ``--meta none`` matches the script — the package wrapper isn't part
    # of what we vendor.
    proc = subprocess.run(
        [
            "uv",
            "run",
            "openapi-python-client",
            "generate",
            "--path",
            str(_OPENAPI_SPEC),
            "--output-path",
            str(work),
            "--meta",
            "none",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, (
        f"openapi-python-client generate failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    committed_files = _walk_relative_files(_GENERATED_DIR)
    regen_files = _walk_relative_files(work)

    only_in_committed = sorted(committed_files - regen_files)
    only_in_regen = sorted(regen_files - committed_files)
    differs: list[Path] = []
    for rel in sorted(committed_files & regen_files):
        if not filecmp.cmp(_GENERATED_DIR / rel, work / rel, shallow=False):
            differs.append(rel)

    if only_in_committed or only_in_regen or differs:
        sections: list[str] = []
        if only_in_committed:
            sections.append(
                "Only in committed _generated/ (delete from tree):\n  "
                + "\n  ".join(str(p) for p in only_in_committed)
            )
        if only_in_regen:
            sections.append(
                "Only in fresh regen (missing from committed):\n  "
                + "\n  ".join(str(p) for p in only_in_regen)
            )
        if differs:
            sections.append("Content differs:\n  " + "\n  ".join(str(p) for p in differs))
        joined = "\n\n".join(sections)
        raise AssertionError(
            "generated SDK is out of date — run scripts/regen-client.sh and "
            "git add packages/aios-sdk/aios_sdk/_generated\n\n" + joined
        )
