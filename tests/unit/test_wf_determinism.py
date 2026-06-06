"""B1.2 — workflow determinism core: canonical_json, content_hash, CallKeyer."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pytest

from aios.workflows.determinism import (
    CallKeyer,
    WorkflowInputTypeError,
    canonical_json,
    content_hash,
)


def test_canonical_json_key_order_invariant() -> None:
    # Insertion order (the PYTHONHASHSEED-adjacent hazard for dicts) must not matter.
    a = canonical_json({"b": 1, "a": 2, "z": {"y": 9, "x": 8}})
    b = canonical_json({"a": 2, "z": {"x": 8, "y": 9}, "b": 1})
    assert a == b == '{"a":2,"b":1,"z":{"x":8,"y":9}}'


def test_canonical_json_accepts_allowed_types() -> None:
    assert canonical_json(None) == "null"
    assert canonical_json(True) == "true"
    assert canonical_json(42) == "42"
    assert canonical_json("hi") == '"hi"'
    assert canonical_json([1, "x", None, {"k": [True]}]) == '[1,"x",null,{"k":[true]}]'


@pytest.mark.parametrize(
    "bad",
    [
        1.0,
        {"n": 1.5},
        [1, 2.0],
        {1, 2},
        (1, 2),
        b"bytes",
        datetime(2026, 1, 1),
        {"when": datetime(2026, 1, 1)},
        object(),
    ],
)
def test_canonical_json_rejects_disallowed_types(bad: object) -> None:
    with pytest.raises(WorkflowInputTypeError):
        canonical_json(bad)


def test_canonical_json_rejects_non_str_dict_keys() -> None:
    with pytest.raises(WorkflowInputTypeError):
        canonical_json({1: "x"})


def test_content_hash_is_sensitive_to_capability_and_spec() -> None:
    base = content_hash("gate", {"q": "ok?"})
    assert base == content_hash("gate", {"q": "ok?"})  # deterministic
    assert base != content_hash("agent", {"q": "ok?"})  # capability_id matters
    assert base != content_hash("gate", {"q": "different"})  # spec matters
    # ...and the "\0" join prevents capability/spec boundary collisions.
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_callkeyer_per_hash_ordinals() -> None:
    keyer = CallKeyer()
    h = content_hash("gate", {"q": "ok?"})
    # Three byte-identical calls disambiguate as #0/#1/#2.
    assert keyer.next("gate", {"q": "ok?"}) == f"sha:{h}#0"
    assert keyer.next("gate", {"q": "ok?"}) == f"sha:{h}#1"
    assert keyer.next("gate", {"q": "ok?"}) == f"sha:{h}#2"
    # A different call has its OWN ordinal stream — divergence stays content-local.
    h2 = content_hash("gate", {"q": "other"})
    assert keyer.next("gate", {"q": "other"}) == f"sha:{h2}#0"
    assert keyer.next("gate", {"q": "ok?"}) == f"sha:{h}#3"


def test_content_hash_stable_across_pythonhashseed() -> None:
    """The same nested input hashes identically under different hash seeds.

    Belt-and-suspenders over the sort_keys + set-ban guarantees: a worker restart
    that randomizes PYTHONHASHSEED must not change any call_key.
    """
    snippet = (
        "from aios.workflows.determinism import content_hash;"
        'print(content_hash("gate", {"b": [1, 2], "a": {"y": 2, "x": 1}, "c": "s"}))'
    )

    def run(seed: str) -> str:
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},  # inherit env, vary only the seed
        )
        return out.stdout.strip()

    assert run("0") == run("123456789")
