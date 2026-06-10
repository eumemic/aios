"""Content-addressed call keys + the determinism contract for workflow scripts.

Replay-with-memo only works if re-running a deterministic script produces the
*same* sequence of capability calls, each with a *stable* key. Two pieces enforce
that:

- :func:`canonical_json` — a total, order-independent serialization over a
  **restricted** input domain. Non-finite floats (NaN/Inf), sets, bytes,
  datetimes, and arbitrary objects are **rejected** at the call site (raising
  :class:`WorkflowInputTypeError`). Finite floats are accepted; integer-valued
  floats (``1.0``, ``2.0``) are normalized to their ``int`` form so ``1`` and
  ``1.0`` produce the same JSON token — the call key is stable even when an
  upstream agent returns ``{"count": 1.0}`` and the script author wrote
  ``{"count": 1}``. Tuples are coerced to lists the same way (JSON has no
  tuple; list is the canonical form). Sets and dicts with non-str keys are
  rejected because an unordered collection has no canonical JSON form (the
  script host pins ``PYTHONHASHSEED=0``, so iteration order is at least stable —
  but stable-arbitrary is not canonical, and the order would still shift across
  interpreter versions). ``sort_keys`` makes dict key order irrelevant.

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
import math
from typing import Any


class WorkflowInputTypeError(TypeError):
    """A capability input contains a type not permitted in a workflow.

    Raised deterministically at the call site (same input → same rejection on
    every replay), so a bad input is a loud author error, never a silent hash
    desync. Allowed: ``None``, ``bool``, ``int``, ``float`` (finite only),
    ``str``, and ``list``/``tuple``/``dict`` of those (tuples canonicalize as
    lists). NaN/Inf floats, sets, bytes, datetimes, and arbitrary objects are
    rejected, as are strings Postgres jsonb cannot store (NUL, unpaired
    surrogates).
    """


def validate_value(x: Any, *, path: str = "input") -> None:
    """Check ``x`` against the workflow value domain (see
    :class:`WorkflowInputTypeError`), raising with a path-precise message on the
    first violation. Capability inputs are validated via :func:`canonical_json`;
    the script host also validates return values directly (same domain, same
    error vocabulary, no canonicalization needed)."""
    if isinstance(x, str):
        # The domain must be storable: Postgres jsonb rejects NUL and lone
        # surrogates, so accepting one here would let the value sail through the
        # host only to detonate at the parent's ::jsonb cast — wedging the run in
        # a deterministic re-wake crashloop instead of a loud author error.
        if "\x00" in x:
            raise WorkflowInputTypeError(f"{path}: NUL (U+0000) is not storable in a workflow")
        try:
            x.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkflowInputTypeError(
                f"{path}: unpaired surrogate is not storable in a workflow"
            ) from exc
        return
    if x is None or isinstance(x, (bool, int)):
        return
    if isinstance(x, float):
        if not math.isfinite(x):
            raise WorkflowInputTypeError(
                f"{path}: non-finite float {x!r} (NaN/Inf have no canonical form)"
            )
        return
    if isinstance(x, (list, tuple)):
        for i, item in enumerate(x):
            validate_value(item, path=f"{path}[{i}]")
        return
    if isinstance(x, dict):
        for key, value in x.items():
            if not isinstance(key, str):
                raise WorkflowInputTypeError(
                    f"{path}: dict keys must be str, got {type(key).__name__}"
                )
            validate_value(value, path=f"{path}.{key}")
        return
    raise WorkflowInputTypeError(
        f"{path}: unsupported workflow input type {type(x).__name__} "
        f"(allowed: None, bool, int, float, str, list, tuple, dict)"
    )


def _canon(x: Any) -> Any:
    """Collapse integer-valued floats (``1.0`` → ``1``) and tuples (→ lists) so
    equal content produces the same JSON token whichever form the author built."""
    if isinstance(x, float):
        return int(x) if x.is_integer() else x
    if isinstance(x, (list, tuple)):
        return [_canon(i) for i in x]
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    return x


def canonical_json(x: Any) -> str:
    """Deterministic, order-independent JSON over the allowed input domain.

    Raises :class:`WorkflowInputTypeError` for any disallowed type *before*
    serializing. Integer-valued floats (``1.0``, ``2.0``) are collapsed to their
    ``int`` form so ``1`` and ``1.0`` produce the same JSON token — the call key is
    content-hash-stable even when an upstream agent returns ``{"count": 1.0}`` and
    the script author wrote ``{"count": 1}``. Tuples are coerced to lists for the
    same reason: ``(1, 2)`` and ``[1, 2]`` key identically.
    """
    validate_value(x)
    return json.dumps(
        _canon(x), separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False
    )


def canonical_schema_json(schema: Any) -> str:
    """Deterministic JSON for a JSON Schema carried in a capability spec.

    Unlike :func:`canonical_json` (which serves the **data** domain and normalises
    integer-valued floats to their ``int`` form), a JSON *Schema* legitimately carries
    decimal numeric constraints (``minimum`` / ``multipleOf`` / …) that must be
    preserved verbatim — normalising ``"minimum": 1.0`` to ``1`` would silently alter
    the schema's meaning. Those are author-written literals that serialize
    deterministically (the ``json.dumps`` float repr is stable for a given value), so
    they are admitted here; ``allow_nan=False`` still rejects the genuinely
    non-deterministic NaN/Inf. ``sort_keys`` keeps the string — and thus the call key —
    independent of the schema's key order.

    ``agent()`` stores this string in the spec so schema floats are preserved verbatim;
    the worker reconstructs the schema with ``json.loads``. Raises ``TypeError`` for a
    non-JSON-serialisable schema (e.g. a set value) — a loud author error, surfaced
    like any other bad ``agent()`` argument.
    """
    return json.dumps(
        schema, separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False
    )


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
