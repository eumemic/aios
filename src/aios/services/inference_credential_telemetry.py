"""Observation surface for explicit migration-policy environment fallback."""

from __future__ import annotations

import contextlib
import threading

from aios.logging import get_logger

log = get_logger(__name__)
_lock = threading.Lock()
_observed_total = 0
try:  # pragma: no cover
    from prometheus_client import Counter

    _FALLBACKS = Counter(
        "aios_inference_legacy_env_fallback_total",
        "Inference calls that fell back to process-environment credentials.",
        ["provider"],
    )
except Exception:  # pragma: no cover
    _FALLBACKS = None


def observe_env_fallback(*, account_id: str, provider: str) -> None:
    """Log and count every fallback admitted by ``observe_legacy_env``."""
    global _observed_total
    with _lock:
        _observed_total += 1
    log.warning(
        "inference_credentials.legacy_env_fallback",
        account_id=account_id,
        provider=provider,
        policy="observe_legacy_env",
    )
    if _FALLBACKS is not None:  # pragma: no cover
        with contextlib.suppress(Exception):
            _FALLBACKS.labels(provider=provider).inc()


def observed_total() -> int:
    with _lock:
        return _observed_total
