"""Deterministic admission over the final provider-bound payload.

Learned token calibration is intentionally absent from this module. Counters are
installed only as reviewed route attestations and bind a route revision to a
provider preflight, exact wire replica, or proved upper bound.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AdmissionMode(StrEnum):
    OBSERVE = "observe"
    ENFORCE = "enforce"


class AdmissionMethod(StrEnum):
    PROVIDER_PREFLIGHT = "provider_preflight"
    EXACT_REPLICA = "exact_replica"
    PROVED_UPPER_BOUND = "proved_upper_bound"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class ProviderPreflight:
    revision: str
    count: Callable[[dict[str, Any]], int]


@dataclass(frozen=True, slots=True)
class ExactCounter:
    revision: str
    count: Callable[[dict[str, Any]], int]


@dataclass(frozen=True, slots=True)
class UpperBoundCounter:
    revision: str
    count: Callable[[dict[str, Any]], int]


Counter = ProviderPreflight | ExactCounter | UpperBoundCounter


@dataclass(frozen=True, slots=True)
class RouteAttestation:
    """Reviewed counting authority for one immutable route revision."""

    route: str
    revision: str
    context_limit: int
    counter: Counter


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    payload_digest: str
    route: str
    route_revision: str | None
    counter_revision: str | None
    method: AdmissionMethod
    bound: int | None
    output_reserve: int | None
    limit: int | None
    verified: bool
    would_reject: bool

    @property
    def total(self) -> int | None:
        if self.bound is None or self.output_reserve is None:
            return None
        return self.bound + self.output_reserve

    def as_event_fields(self) -> dict[str, object]:
        return {
            "context_admission_payload_digest": self.payload_digest,
            "context_admission_route": self.route,
            "context_admission_route_revision": self.route_revision,
            "context_admission_counter_revision": self.counter_revision,
            "context_admission_method": self.method.value,
            "context_admission_bound": self.bound,
            "context_admission_output_reserve": self.output_reserve,
            "context_admission_limit": self.limit,
            "context_admission_verified": self.verified,
            "context_admission_would_reject": self.would_reject,
        }


class ContextAdmissionRejected(Exception):
    def __init__(self, report: AdmissionReport, reason: str) -> None:
        super().__init__(reason)
        self.report = report


def payload_digest(payload: Mapping[str, Any], *, route_revision: str | None) -> str:
    """Bind a canonical final payload to its exact route revision."""
    canonical = json.dumps(
        {"payload": payload, "route_revision": route_revision},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", "surrogatepass")
    return hashlib.sha256(canonical).hexdigest()


def _output_reserve(payload: Mapping[str, Any]) -> int | None:
    for key in ("max_output_tokens", "max_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _method(counter: Counter) -> AdmissionMethod:
    if isinstance(counter, ProviderPreflight):
        return AdmissionMethod.PROVIDER_PREFLIGHT
    if isinstance(counter, ExactCounter):
        return AdmissionMethod.EXACT_REPLICA
    return AdmissionMethod.PROVED_UPPER_BOUND


def admit_context(
    payload: dict[str, Any],
    *,
    mode: AdmissionMode,
    attestation: RouteAttestation | None,
) -> AdmissionReport:
    """Observe or reject using only deterministic, route-bound authority.

    The caller must pass the final object that it submits next. No learned
    estimate can enter this API. Missing attestation or output cap is
    unverified and therefore fail-closed in enforce mode.
    """
    route = str(payload.get("model", ""))
    verified_attestation = (
        attestation if attestation is not None and attestation.route == route else None
    )
    revision = verified_attestation.revision if verified_attestation is not None else None
    digest = payload_digest(payload, route_revision=revision)
    reserve = _output_reserve(payload)

    if verified_attestation is None or reserve is None:
        report = AdmissionReport(
            payload_digest=digest,
            route=route,
            route_revision=revision,
            counter_revision=None,
            method=AdmissionMethod.UNVERIFIED,
            bound=None,
            output_reserve=reserve,
            limit=(
                verified_attestation.context_limit if verified_attestation is not None else None
            ),
            verified=False,
            would_reject=True,
        )
        if mode is AdmissionMode.ENFORCE:
            reason = "no verified counter for final payload route"
            if verified_attestation is not None:
                reason = "final payload has no enforced output token cap"
            raise ContextAdmissionRejected(report, reason)
        return report

    attestation = verified_attestation
    bound = attestation.counter.count(payload)
    if bound < 0:
        raise ValueError("context counter returned a negative bound")
    total = bound + reserve
    report = AdmissionReport(
        payload_digest=digest,
        route=route,
        route_revision=attestation.revision,
        counter_revision=attestation.counter.revision,
        method=_method(attestation.counter),
        bound=bound,
        output_reserve=reserve,
        limit=attestation.context_limit,
        verified=True,
        would_reject=total > attestation.context_limit,
    )
    if report.would_reject and mode is AdmissionMode.ENFORCE:
        raise ContextAdmissionRejected(
            report,
            f"context admission rejected {total} tokens against route limit {attestation.context_limit}",
        )
    return report


# Empty until a route's exact counter and revision pin pass independent review.
_ROUTE_ATTESTATIONS: dict[str, RouteAttestation] = {}


def route_attestation(route: str) -> RouteAttestation | None:
    return _ROUTE_ATTESTATIONS.get(route)
