"""Supply-chain contract for the channel connector images (signal / whatsapp / telegram).

These images used to ``uv pip install --no-sources`` their packages, which
resolved every dependency fresh from PyPI at build time — nothing tied the
shipped versions to uv.lock.  The contract now: Python deps come from a
``uv sync --frozen`` scoped to the connector's workspace member, and every
external image (``FROM`` and ``COPY --from``) is pinned by digest.

The Dockerfiles are read as text (not parsed) — the contract is "this
instruction is present / absent", which is what a regression would silently
change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
CHANNELS = ("signal", "whatsapp", "telegram")
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _instructions(channel: str) -> list[str]:
    text = (REPO / "connectors" / channel / "Dockerfile").read_text()
    # Fold line continuations; drop comments (the signal file quotes a stock
    # install snippet in its REVERT-TO-STOCK marker).
    folded = text.replace("\\\n", " ")
    return [
        line.strip()
        for line in folded.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("channel", CHANNELS)
def test_installs_from_lock_scoped_to_its_member(channel: str) -> None:
    runs = [line for line in _instructions(channel) if line.startswith("RUN ")]
    syncs = [line for line in runs if "uv sync" in line]
    assert syncs, runs
    for sync in syncs:
        assert "--frozen" in sync, sync
        assert "--no-dev" in sync, sync
        # Scoped to the member: an unscoped sync would install the root
        # ``aios`` project (litellm & co.) into the connector image.
        assert f"--package aios-{channel}" in sync, sync


@pytest.mark.parametrize("channel", CHANNELS)
def test_no_unlocked_pip_installs(channel: str) -> None:
    for line in _instructions(channel):
        assert "pip install" not in line, line
        assert "--no-sources" not in line, line


@pytest.mark.parametrize("channel", CHANNELS)
def test_venv_is_on_path(channel: str) -> None:
    # entrypoint.sh / HEALTHCHECK / CMD run bare ``python``.
    env_lines = [line for line in _instructions(channel) if line.startswith("ENV ")]
    assert any("UV_PROJECT_ENVIRONMENT=/app/.venv" in line for line in env_lines), env_lines
    assert any('PATH="/app/.venv/bin:' in line for line in env_lines), env_lines


@pytest.mark.parametrize("channel", CHANNELS)
def test_external_images_are_digest_pinned(channel: str) -> None:
    lines = _instructions(channel)
    # ``FROM [--platform=...] <image> [AS <stage>]`` — skip flags to reach the image.
    from_args = [
        [arg for arg in line.split()[1:] if not arg.startswith("--")]
        for line in lines
        if line.startswith("FROM ")
    ]
    stages = {args[2] for args in from_args if len(args) == 3 and args[1].upper() == "AS"}
    from_images = [args[0] for args in from_args]
    copy_from = [m.group(1) for line in lines if (m := re.search(r"^COPY .*--from=(\S+)", line))]
    external = [ref for ref in [*from_images, *copy_from] if ref not in stages]
    assert external
    for ref in external:
        assert DIGEST.search(ref), f"{ref} is not pinned by digest"
