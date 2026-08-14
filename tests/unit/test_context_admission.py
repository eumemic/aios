"""Deterministic final-payload context admission."""

from __future__ import annotations

import pytest

from aios.harness.context_admission import (
    AdmissionMethod,
    AdmissionMode,
    ContextAdmissionRejected,
    ExactCounter,
    RouteAttestation,
    admit_context,
    payload_digest,
)


def _attestation(*, revision: str = "r1") -> RouteAttestation:
    return RouteAttestation(
        route="test/model",
        revision=revision,
        context_limit=10,
        counter=ExactCounter(
            revision="tokenizer-1",
            count=lambda payload: len(payload["messages"][0]["content"]),
        ),
    )


def test_exact_boundary_c_is_admitted_and_c_plus_one_is_rejected() -> None:
    attestation = _attestation()
    at_limit = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "12345678"}],
        "max_tokens": 2,
    }
    over_limit = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "123456789"}],
        "max_tokens": 2,
    }

    report = admit_context(at_limit, mode=AdmissionMode.ENFORCE, attestation=attestation)
    assert report.bound == 8
    assert report.output_reserve == 2
    assert report.method is AdmissionMethod.EXACT_REPLICA

    with pytest.raises(ContextAdmissionRejected) as exc_info:
        admit_context(over_limit, mode=AdmissionMode.ENFORCE, attestation=attestation)
    assert exc_info.value.report.total == 11


def test_observe_mode_never_changes_runtime_behavior() -> None:
    report = admit_context(
        {"model": "unknown", "messages": [{"role": "user", "content": "too large"}]},
        mode=AdmissionMode.OBSERVE,
        attestation=None,
    )
    assert report.verified is False
    assert report.would_reject is True


def test_enforce_fails_closed_for_unknown_route() -> None:
    with pytest.raises(ContextAdmissionRejected, match="no verified counter"):
        admit_context(
            {"model": "unknown", "messages": []},
            mode=AdmissionMode.ENFORCE,
            attestation=None,
        )


def test_digest_binds_payload_and_route_revision() -> None:
    payload = {"model": "test/model", "messages": [{"role": "user", "content": "same"}]}
    first = admit_context(payload, mode=AdmissionMode.OBSERVE, attestation=_attestation())
    mutated = admit_context(
        {**payload, "messages": [{"role": "user", "content": "changed"}]},
        mode=AdmissionMode.OBSERVE,
        attestation=_attestation(),
    )
    revised = admit_context(
        payload, mode=AdmissionMode.OBSERVE, attestation=_attestation(revision="r2")
    )

    assert first.payload_digest != mutated.payload_digest
    assert first.payload_digest != revised.payload_digest
    assert first.payload_digest == payload_digest(payload, route_revision="r1")


def test_tools_multimodal_and_framing_reach_exact_counter_unchanged() -> None:
    seen: list[dict[str, object]] = []
    payload = {
        "model": "test/model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ],
            }
        ],
        "tools": [
            {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        ],
        "extra_body": {"provider_framing": "v1"},
        "max_tokens": 10,
    }

    def count(final: dict[str, object]) -> int:
        seen.append(final)
        return 5

    attestation = RouteAttestation(
        route="test/model",
        revision="wire-v1",
        context_limit=100,
        counter=ExactCounter(revision="tok-v1", count=count),
    )

    admit_context(payload, mode=AdmissionMode.OBSERVE, attestation=attestation)
    assert seen == [payload]
