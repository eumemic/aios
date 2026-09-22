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


@pytest.mark.parametrize("spelling", ["max_tokens", "max_output_tokens", "max_completion_tokens"])
def test_every_accepted_cap_spelling_is_a_verified_output_reserve(spelling: str) -> None:
    """REQUIRED TEST — a request capped under ANY accepted spelling is admitted
    with a VERIFIED reserve equal to that value, in observe mode and under
    enforcement.

    ``max_completion_tokens`` is the case that was red: ``completion.py`` accepts
    it as an explicit cap (so the harness injects no ``max_tokens`` of its own)
    and litellm maps it onto the provider's native spelling, so a real cap went
    on the wire — but ``_output_reserve`` read only two of the three spellings,
    so admission called the payload ``unverified``/``would_reject`` in observe
    mode and raised "no enforced output token cap" under enforcement. The
    parametrization is deliberate: pinning the three together is what stops the
    list drifting again one spelling at a time.
    """
    attestation = _attestation()
    payload = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "1234"}],
        spelling: 3,
    }

    observed = admit_context(payload, mode=AdmissionMode.OBSERVE, attestation=attestation)
    assert observed.output_reserve == 3
    assert observed.verified is True
    assert observed.method is AdmissionMethod.EXACT_REPLICA
    assert observed.bound == 4
    assert observed.total == 7  # 4 + 3, under the limit of 10
    assert observed.would_reject is False

    # And under enforcement it is admitted rather than raising.
    enforced = admit_context(payload, mode=AdmissionMode.ENFORCE, attestation=attestation)
    assert enforced.verified is True
    assert enforced.output_reserve == 3


@pytest.mark.parametrize("spelling", ["max_tokens", "max_output_tokens", "max_completion_tokens"])
def test_every_accepted_cap_spelling_counts_toward_the_limit(spelling: str) -> None:
    """The reserve is not merely *recorded* per spelling — it is CHARGED against
    the route limit identically. A ``_output_reserve`` that returned the value
    but excluded it from ``total`` would pass the test above and still admit an
    over-limit payload, so pin the rejection side per spelling too.
    """
    attestation = _attestation()  # context_limit=10
    payload = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "12345678"}],  # bound = 8
        spelling: 3,  # 8 + 3 = 11 > 10
    }

    observed = admit_context(payload, mode=AdmissionMode.OBSERVE, attestation=attestation)
    assert observed.total == 11
    assert observed.would_reject is True

    with pytest.raises(ContextAdmissionRejected, match="context admission rejected 11"):
        admit_context(payload, mode=AdmissionMode.ENFORCE, attestation=attestation)


def test_admission_reads_the_shared_cap_spelling_list() -> None:
    """The three surfaces that must agree on the accepted spellings read ONE
    list, so none can fall behind the others again.

    This is the actual defect class: ``max_completion_tokens`` was added to
    ``completion``'s injection gate and to ``output_reservation`` while the
    admission gate kept its own inline pair. Asserting identity (``is``) rather
    than equality means re-introducing a private copy fails here even if it
    happens to be correct on the day it is written.
    """
    from aios.harness import completion, context_admission
    from aios.harness.context_budget import EXPLICIT_OUTPUT_CAP_KEYS

    assert context_admission.EXPLICIT_OUTPUT_CAP_KEYS is EXPLICIT_OUTPUT_CAP_KEYS
    assert completion.EXPLICIT_OUTPUT_CAP_KEYS is EXPLICIT_OUTPUT_CAP_KEYS
    assert set(EXPLICIT_OUTPUT_CAP_KEYS) == {
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
    }


def test_an_uncapped_payload_is_still_unverified() -> None:
    """The negative half. Widening the spelling list must not turn "no cap named"
    into a verified reserve — fail-closed on an uncapped payload is the property
    the admission gate exists for, and a ``_output_reserve`` that returned some
    default instead of ``None`` would pass every test above.
    """
    payload = {"model": "test/model", "messages": [{"role": "user", "content": "1234"}]}

    report = admit_context(payload, mode=AdmissionMode.OBSERVE, attestation=_attestation())
    assert report.output_reserve is None
    assert report.verified is False
    assert report.would_reject is True

    with pytest.raises(ContextAdmissionRejected, match="no enforced output token cap"):
        admit_context(payload, mode=AdmissionMode.ENFORCE, attestation=_attestation())


def test_a_zero_or_bool_cap_is_not_a_cap_under_any_spelling() -> None:
    """``max_completion_tokens: 0`` and ``: True`` must not be read as caps of 0
    or 1 — the same guard the other two spellings already had. (``bool`` is an
    ``int`` subclass in Python, which is why the explicit check exists.)
    """
    for spelling in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
        for bad in (0, -5, True, "4096", None):
            report = admit_context(
                {
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "1234"}],
                    spelling: bad,
                },
                mode=AdmissionMode.OBSERVE,
                attestation=_attestation(),
            )
            assert report.output_reserve is None, f"{spelling}={bad!r} was read as a cap"
            assert report.verified is False
