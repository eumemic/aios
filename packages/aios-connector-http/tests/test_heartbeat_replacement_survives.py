"""Reproduction + guards for PR #2355 finding F2.

Property: Heartbeat claim and refresh must never unlink -- or move into staging
and destroy -- any inode they did not create or verify under lock. An
independent replacement placed at the heartbeat path must survive intact.

Failure mode: the check-then-exchange uses the shared public pathname with no
guard. If the path is replaced around the ``RENAME_EXCHANGE`` the replacement is
swept into the private staging directory, and the ``finally`` cleanup then
``unlink``s it -- destroying an operator file the process never created.

Two threat models are covered:

* SINGLE injection (the realistic operator-replaces-once case): the replacement
  must end up back at the PUBLIC path, intact. This is the full property.
* DOUBLE injection (the reviewer's adversarial case: a fresh replacement dropped
  before EVERY exchange, including the rollback): every injected inode must
  SURVIVE somewhere on disk. Placing the very last one at the public path is
  impossible against an adversary that replaces during the final exchange, but
  the F2 defect -- silently *deleting* it -- must not happen.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aios_connector_http.runner as runner_mod
from aios_connector_http.runner import HttpConnector


def _payload(healthy: list[str], unhealthy: list[str]) -> bytes:
    return json.dumps(
        {
            "healthy_connection_ids": healthy,
            "unhealthy_connection_ids": unhealthy,
        },
        sort_keys=True,
    ).encode()


def _inject_before_exchanges(path: Path, replacements: list[bytes]) -> Callable[[Any, Any], bool]:
    """Return a ``_rename_exchange`` wrapper that drops a new operator inode at
    ``path`` immediately before the first ``len(replacements)`` real exchanges,
    mimicking an operator atomically replacing the heartbeat pathname at the
    worst possible instant."""
    real = runner_mod._rename_exchange
    state = {"n": 0}

    def _wrapped(source: Any, destination: Any) -> bool:
        idx = state["n"]
        state["n"] += 1
        if idx < len(replacements):
            tmp = path.parent / f".operator_repl_{idx}"
            tmp.write_bytes(replacements[idx])
            os.replace(tmp, path)
        return real(source, destination)

    return _wrapped


def _all_bytes_on_disk(root: Path) -> list[bytes]:
    found: list[bytes] = []
    for p in root.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                found.append(p.read_bytes())
    return found


# ---------------------------------------------------------------------------
# SINGLE injection -- the realistic property: replacement survives AT the path.
# ---------------------------------------------------------------------------


def test_refresh_single_replacement_survives_at_public_path(
    monkeypatch: Any, tmp_path: Path
) -> None:
    path = tmp_path / "hb"
    first = _payload(["conn_1"], [])
    identity = HttpConnector._claim_heartbeat(path, first, True)
    assert identity is not None

    replacement = b"operator replacement"
    monkeypatch.setattr(
        runner_mod, "_rename_exchange", _inject_before_exchanges(path, [replacement])
    )

    second = _payload(["conn_1", "conn_2"], [])
    HttpConnector._refresh_heartbeat(path, identity, second, True)

    assert path.exists(), "the heartbeat path was left with no inode at all"
    assert path.read_bytes() == replacement, (
        f"operator replacement did not survive at the public path; path holds {path.read_bytes()!r}"
    )


def test_claim_single_replacement_survives_at_public_path(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "hb"
    stale = HttpConnector._claim_heartbeat(path, _payload([], ["conn_1"]), False)
    assert stale is not None

    replacement = b"operator replacement"
    monkeypatch.setattr(
        runner_mod, "_rename_exchange", _inject_before_exchanges(path, [replacement])
    )

    HttpConnector._claim_heartbeat(path, _payload(["conn_1"], []), True)

    assert path.exists(), "the heartbeat path was left with no inode at all"
    assert path.read_bytes() == replacement, (
        f"operator replacement did not survive at the public path; path holds {path.read_bytes()!r}"
    )


# ---------------------------------------------------------------------------
# DOUBLE injection -- the reviewer's adversarial case: nothing is DELETED.
# ---------------------------------------------------------------------------


def test_refresh_double_injection_destroys_nothing(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "hb"
    first = _payload(["conn_1"], [])
    identity = HttpConnector._claim_heartbeat(path, first, True)
    assert identity is not None

    replacements = [b"operator replacement 1", b"operator replacement 2"]
    monkeypatch.setattr(
        runner_mod, "_rename_exchange", _inject_before_exchanges(path, replacements)
    )

    second = _payload(["conn_1", "conn_2"], [])
    HttpConnector._refresh_heartbeat(path, identity, second, True)

    on_disk = _all_bytes_on_disk(tmp_path)
    for r in replacements:
        assert r in on_disk, (
            f"operator replacement {r!r} was DESTROYED by cleanup; surviving inodes: {on_disk!r}"
        )
    assert path.exists(), "the public heartbeat path was left with no inode at all"


def test_claim_double_injection_destroys_nothing(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "hb"
    stale = HttpConnector._claim_heartbeat(path, _payload([], ["conn_1"]), False)
    assert stale is not None

    replacements = [b"operator replacement A", b"operator replacement B"]
    monkeypatch.setattr(
        runner_mod, "_rename_exchange", _inject_before_exchanges(path, replacements)
    )

    HttpConnector._claim_heartbeat(path, _payload(["conn_1"], []), True)

    on_disk = _all_bytes_on_disk(tmp_path)
    for r in replacements:
        assert r in on_disk, (
            f"operator replacement {r!r} was DESTROYED by cleanup; surviving inodes: {on_disk!r}"
        )
    assert path.exists(), "the public heartbeat path was left with no inode at all"


# ---------------------------------------------------------------------------
# OVER-CORRECTION GUARD -- the fix must still publish in the unraced case.
# ---------------------------------------------------------------------------


def test_refresh_without_race_still_publishes(tmp_path: Path) -> None:
    """The degenerate 'never exchange / never clean up' fix would also make the
    destroy-nothing tests pass. Assert the normal path still publishes the new
    snapshot AND leaves no staging leak."""
    path = tmp_path / "hb"
    first = _payload(["conn_1"], [])
    identity = HttpConnector._claim_heartbeat(path, first, True)
    assert identity is not None

    second = _payload(["conn_1", "conn_2"], [])
    new_identity = HttpConnector._refresh_heartbeat(path, identity, second, True)

    assert new_identity is not None, "refresh refused a legitimate, unraced publish"
    assert path.read_bytes() == second
    # No staging directory or claimant may be left behind on the happy path.
    leftovers = [p for p in tmp_path.iterdir() if p.name != "hb"]
    assert leftovers == [], f"staging leak on the happy path: {leftovers!r}"


def test_claim_stale_without_race_still_reclaims(tmp_path: Path) -> None:
    path = tmp_path / "hb"
    stale = HttpConnector._claim_heartbeat(path, _payload([], ["conn_1"]), False)
    assert stale is not None

    healthy = _payload(["conn_1"], [])
    identity = HttpConnector._claim_heartbeat(path, healthy, True)

    assert identity is not None, "stale reclaim refused a legitimate, unraced claim"
    assert path.read_bytes() == healthy
    leftovers = [p for p in tmp_path.iterdir() if p.name != "hb"]
    assert leftovers == [], f"staging leak on the happy path: {leftovers!r}"
