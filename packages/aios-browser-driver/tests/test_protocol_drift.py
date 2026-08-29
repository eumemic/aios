"""The vendored browser_protocol.py must never drift from the src/ original.

ONE module, TWO consumers (the aios worker in-process; the browser image via a
build-time COPY). This test is the drift guard: it fails the moment the
package's vendored copy stops being byte-identical to
``src/aios/sandbox/browser_protocol.py``.
"""

from __future__ import annotations

from pathlib import Path


def test_protocol_copy_is_byte_identical() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    original = repo_root / "src" / "aios" / "sandbox" / "browser_protocol.py"
    copy = (
        repo_root
        / "packages"
        / "aios-browser-driver"
        / "aios_browser_driver"
        / "browser_protocol.py"
    )
    assert copy.read_bytes() == original.read_bytes(), (
        "browser_protocol.py drifted from src/aios/sandbox/browser_protocol.py — "
        "edit the src/ original and re-copy it verbatim:\n"
        "  cp src/aios/sandbox/browser_protocol.py "
        "packages/aios-browser-driver/aios_browser_driver/browser_protocol.py"
    )
