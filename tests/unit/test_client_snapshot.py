"""Drift check: regenerating the client from ``openapi.json`` must match
the committed ``packages/aios-sdk/aios_sdk/_generated/`` tree byte-for-byte.

Companion to :mod:`tests.unit.test_openapi_snapshot`. That test keeps
``openapi.json`` honest against the live FastAPI app; this one keeps the
generated client honest against ``openapi.json``. Together they close
the loop spec → client.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import aios_sdk


def test_generated_client_matches_committed(tmp_path: Path) -> None:
    committed = Path(aios_sdk.__file__).parent / "_generated"
    openapi_path = Path(__file__).resolve().parents[2] / "openapi.json"

    subprocess.run(
        [
            "openapi-python-client",
            "generate",
            "--path",
            str(openapi_path),
            "--output-path",
            str(tmp_path),
            "--meta",
            "none",
            "--overwrite",
        ],
        check=True,
    )

    committed_files = {p.relative_to(committed) for p in committed.rglob("*.py")}
    generated_files = {p.relative_to(tmp_path) for p in tmp_path.rglob("*.py")}

    missing = sorted(committed_files - generated_files)
    extra = sorted(generated_files - committed_files)
    assert not missing and not extra, (
        f"_generated/ file set drifted from spec.\n"
        f"  only in committed: {missing}\n"
        f"  only in generated: {extra}\n"
        f"Run scripts/regen-client.sh and commit the result."
    )

    diffs = [
        str(rel)
        for rel in sorted(committed_files)
        if (committed / rel).read_bytes() != (tmp_path / rel).read_bytes()
    ]
    assert not diffs, (
        f"_generated/ contents drifted from spec — {len(diffs)} files differ:\n"
        f"  {diffs[:10]}\n"
        f"Run scripts/regen-client.sh and commit the result."
    )
