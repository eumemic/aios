"""Content-addressed call keys + the determinism contract for workflow scripts.

Replay-with-memo only works if re-running a deterministic script produces the
*same* sequence of capability calls, each with a *stable* key. Two pieces enforce
that:

- :func:`canonical_json` — a total, order-independent serialization over a
  **restricted** input domain. Floats, sets, tuples, bytes, datetimes, and
  arbitrary objects are **rejected** at the call site (raising
  :class:`WorkflowInputTypeError`) rather than silently coerced — these are
  exactly the types whose serialization is ambiguous (``1.0`` vs ``1``) or whose
  iteration order is ``PYTHONHASHSEED``-sensitive (``set``), which would desync
  the key across a worker restart. ``sort_keys`` makes dict key order
  irrelevant.

- :class:`CallKeyer` — turns ``(capability_id, input)`` into
  ``"sha:<hex>#<n>"`` where ``n`` is the count of *prior* emissions in this run
  that share the same content hash. The disambiguator is **per-content-hash**,
  not a global counter, so an extra/missing upstream call shifts only its own
  hash's ordinals — divergence stays content-local instead of cascading down the
  whole suffix.

Stdlib-only (this module is imported by the credential-free script-host
subprocess).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class WorkflowInputTypeError(TypeError):
    """A capability input contains a type not permitted in a workflow.

    Raised deterministically at the call site (same input → same rejection on
    every replay), so a bad input is a loud author error, never a silent hash
    desync. Allowed: ``None``, ``bool``, ``int``, ``str``, and ``list``/``dict``
    of those. Authors needing a float pass a string or scaled int; a set, a
    sorted list.
    """


def _validate(x: Any, *, path: str = "input") -> None:
    if x is None or isinstance(x, (bool, int, str)):
        return
    if isinstance(x, list):
        for i, item in enumerate(x):
            _validate(item, path=f"{path}[{i}]")
        return
    if isinstance(x, dict):
        for key, value in x.items():
            if not isinstance(key, str):
                raise WorkflowInputTypeError(
                    f"{path}: dict keys must be str, got {type(key).__name__}"
                )
            _validate(value, path=f"{path}.{key}")
        return
    raise WorkflowInputTypeError(
        f"{path}: unsupported workflow input type {type(x).__name__} "
        f"(allowed: None, bool, int, str, list, dict)"
    )


def canonical_json(x: Any) -> str:
    """Deterministic, order-independent JSON over the allowed input domain.

    Raises :class:`WorkflowInputTypeError` for any disallowed type *before*
    serializing.
    """
    _validate(x)
    return json.dumps(x, separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False)


def content_hash(capability_id: str, spec: Any) -> str:
    """``sha256(capability_id ‖ "\\0" ‖ canonical_json(spec))`` as hex."""
    payload = f"{capability_id}\0{canonical_json(spec)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CallKeyer:
    """Allocates content-addressed call keys with per-content-hash ordinals.

    A fresh keyer is created at the start of every wake's replay. Because replay
    re-runs the same deterministic emission sequence, it yields the identical key
    sequence each time — the basis of replay-with-memo. Not thread-safe (the
    driver is single-threaded).
    """

    def __init__(self) -> None:
        self._ordinals: dict[str, int] = {}

    def next(self, capability_id: str, spec: Any) -> str:
        h = content_hash(capability_id, spec)
        ordinal = self._ordinals.get(h, 0)
        self._ordinals[h] = ordinal + 1
        return f"sha:{h}#{ordinal}"
