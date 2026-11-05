"""Undelivered-attachment marker helpers (#2551).

A dropped inbound attachment must reach the agent as a one-line marker
in the inbound content instead of vanishing into a log line.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path
from typing import Any

import pytest
import structlog

from aios_connector_http import sandbox
from aios_connector_http.sandbox import (
    FALLBACK_MARKER,
    MAX_ATTACHMENT_BYTES,
    Attachment,
    AttachmentError,
    _emit,
    append_markers,
    coerce_size,
    drop_attachment,
    format_size,
    is_failed_download,
    is_over_cap,
    rejection_marker,
    safe_marker,
)

ISSUE_MARKER = (
    "[attachment not delivered: 'Online FSOT Practice Test.pdf', "
    "application/pdf, 7.1 MiB > 5 MiB limit]"
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (10, "10 B"),
        (1023, "1023 B"),
        (1024, "1 KiB"),
        (2048, "2 KiB"),
        (1048576, "1 MiB"),
        (5242880, "5 MiB"),
        (7441443, "7.1 MiB"),
        (8153223, "7.8 MiB"),
        (163815482, "156.2 MiB"),
        (3 * 2**30, "3 GiB"),
    ],
)
def test_format_size(n: int, expected: str) -> None:
    assert format_size(n) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7441443", 7441443),
        (7441443, 7441443),
        (0, 0),
        ("0", 0),
        (None, None),
        (-1, None),
        (True, None),
        (False, None),
        ("abc", None),
        ("", None),
        (" 5", None),
        ("+5", None),
        ("-5", None),
        ("\u00b2", None),
        ("\u0661\u0662", None),
        (1.5, None),
        (2.0, None),
        (float("nan"), None),
        (object(), None),
    ],
)
def test_coerce_size(value: object, expected: int | None) -> None:
    assert coerce_size(value) == expected


@pytest.mark.parametrize(
    ("st", "declared", "expected"),
    [
        (0, 163815482, True),
        (0, None, False),
        (0, 0, False),
        (5, 163815482, False),
        (None, 163815482, False),
    ],
)
def test_is_failed_download(st: int | None, declared: int | None, expected: bool) -> None:
    assert is_failed_download(st, declared) is expected


def test_is_over_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    assert is_over_cap(None) is False
    assert is_over_cap(MAX_ATTACHMENT_BYTES) is False
    assert is_over_cap(MAX_ATTACHMENT_BYTES + 1) is True
    monkeypatch.setattr(sandbox, "MAX_ATTACHMENT_BYTES", 1024 * 1024)
    assert is_over_cap(1048577) is True


def test_marker_variants_exact() -> None:
    assert (
        rejection_marker("Online FSOT Practice Test.pdf", "application/pdf", 7441443)
        == ISSUE_MARKER
    )
    assert (
        rejection_marker("x.mp4", "video/mp4", 163815482, failed_download=True)
        == "[attachment not delivered: 'x.mp4', video/mp4, 156.2 MiB > 5 MiB limit: download failed]"
    )
    assert (
        rejection_marker("x.pdf", "application/pdf", 1258291, failed_download=True)
        == "[attachment not delivered: 'x.pdf', application/pdf, 1.2 MiB: download failed]"
    )
    assert (
        rejection_marker("x.pdf", "application/pdf", None, failed_download=True)
        == "[attachment not delivered: 'x.pdf', application/pdf: download failed]"
    )
    assert (
        rejection_marker("x.pdf", "application/pdf", 1258291)
        == "[attachment not delivered: 'x.pdf', application/pdf, 1.2 MiB: rejected]"
    )
    assert (
        rejection_marker("x.pdf", "application/pdf", None)
        == "[attachment not delivered: 'x.pdf', application/pdf: rejected]"
    )


def test_marker_explicit_over_cap_false_wins() -> None:
    assert rejection_marker("x.pdf", "application/pdf", 7441443, over_cap=False).endswith(
        "7.1 MiB: rejected]"
    )


def test_marker_reads_cap_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "MAX_ATTACHMENT_BYTES", 1024 * 1024)
    assert rejection_marker("x", "application/pdf", 2 * 1024 * 1024).endswith(
        "2 MiB > 1 MiB limit]"
    )


def test_no_contradiction_at_limit_plus_one() -> None:
    assert rejection_marker("x", "application/pdf", MAX_ATTACHMENT_BYTES + 1).endswith(
        "5,242,881 bytes > 5,242,880 bytes limit]"
    )
    for size in range(MAX_ATTACHMENT_BYTES + 1, MAX_ATTACHMENT_BYTES + 100000, 997):
        marker = rejection_marker("x", "application/pdf", size)
        lhs, rhs = marker.split(", ")[-1].rstrip("]").split(" > ")
        assert lhs != rhs.removesuffix(" limit")


def test_no_marker_says_too_large() -> None:
    variants = [
        rejection_marker("a", "application/pdf", s, failed_download=f)
        for s in (None, 10, 7441443, 163815482)
        for f in (False, True)
    ]
    assert all("too large" not in v for v in variants)


def test_marker_never_raises() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert rejection_marker(Hostile(), object(), "garbage").startswith(  # type: ignore[arg-type]
        "[attachment not delivered: 'unnamed', unknown type"
    )


_LOOKALIKES = set("'\"[]\u2018\u2019\u201a\u201b\u201c\u201d\u2039\u203a\u3010\u3011\u300c\u300d")


@pytest.mark.parametrize(
    "name",
    [
        "a'] \n[evil",
        "x\u2028y",
        "x\u2029y",
        "x\u0085y",
        "\u202efdp.exe",
        "x\u2066y",
        "x\u200by",
        "\ufeffx",
        "\uff3d",
        "\uff07",
        "\u2019",
    ],
)
def test_safe_marker_injection(name: str) -> None:
    marker = rejection_marker(name, "application/pdf", 7441443)
    assert len(marker.splitlines()) == 1
    assert all(unicodedata.category(ch) not in {"Cc", "Cf", "Zl", "Zp"} for ch in marker)
    safe_name, _ = safe_marker(name, "application/pdf")
    assert not (set(safe_name) & _LOOKALIKES)


def test_safe_marker_bounds_and_defaults() -> None:
    long_name, _ = safe_marker("\u00e9" * 500, None)
    assert len(long_name) == 120 and long_name.endswith("\u2026")
    assert safe_marker("", None) == ("unnamed", "unknown type")
    assert safe_marker(None, None)[0] == "unnamed"
    assert safe_marker("x", "text/html\n")[1] == "unknown type"
    assert safe_marker("x", "a/b/c")[1] == "unknown type"
    assert safe_marker("x", "a]b/c")[1] == "unknown type"
    assert safe_marker("x", "a/" + "b" * 200)[1] == "unknown type"
    assert safe_marker("x", "application/pdf; charset=x")[1] == "application/pdf"


def test_append_markers() -> None:
    assert append_markers("hi", []) == "hi"
    assert append_markers("", []) == ""
    assert append_markers(None, []) == ""
    assert append_markers("", ["m1"]) == "m1"
    assert append_markers(None, ["m1", "m2"]) == "m1\nm2"
    assert append_markers("hi", ["m1", "m2"]) == "hi\nm1\nm2"


def test_marker_deterministic() -> None:
    a = rejection_marker("x.pdf", "application/pdf", 163815482, failed_download=True)
    b = rejection_marker("x.pdf", "application/pdf", 163815482, failed_download=True)
    assert a == b


def test_as_params_over_cap_message_unchanged(tmp_path: Path) -> None:
    p = tmp_path / "big.pdf"
    with open(p, "wb") as f:
        f.truncate(7441443)
    with pytest.raises(AttachmentError) as exc:
        Attachment(host_path=str(p), filename="big.pdf", content_type="application/pdf").as_params()
    assert str(exc.value) == (
        "attachment 'big.pdf' is 7441443 bytes; SDK cap is 5242880 bytes (5 MiB)."
    )
    assert os.path.getsize(p) == 7441443


def test_emit_with_stdlib_logger(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("t.markers")
    with caplog.at_level(logging.WARNING, logger="t.markers"):
        _emit(logger, "evt", reason="over_cap", attachment_name="x", marker_emitted=True)
    [rec] = [r for r in caplog.records if r.name == "t.markers"]
    assert rec.getMessage() == "evt"
    assert rec.__dict__["attachment_name"] == "x"


def test_emit_with_structlog_logger() -> None:
    with structlog.testing.capture_logs() as logs:
        _emit(structlog.get_logger("t"), "evt", reason="over_cap", marker_emitted=True)
    assert logs == [
        {"event": "evt", "reason": "over_cap", "marker_emitted": True, "log_level": "warning"}
    ]


def test_emit_with_raising_logger_does_not_raise() -> None:
    class Bad:
        def warning(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("logger broken")

    _emit(Bad(), "evt", reason="x")


def test_drop_attachment_appends_and_logs() -> None:
    markers: list[str] = []
    with structlog.testing.capture_logs() as logs:
        drop_attachment(
            markers,
            structlog.get_logger("t"),
            "signal",
            name="Online FSOT Practice Test.pdf",
            content_type="application/pdf",
            size=7441443,
            reason="over_cap",
        )
    assert markers == [ISSUE_MARKER]
    [entry] = logs
    assert entry["event"] == "signal.inbound.attachment_not_delivered"
    assert entry["reason"] == "over_cap"
    assert entry["attachment_name"] == "Online FSOT Practice Test.pdf"


def test_drop_attachment_builder_failure_uses_fallback_and_logs() -> None:
    def broken(*a: Any, **k: Any) -> str:
        raise RuntimeError("builder bug")

    markers: list[str] = []
    with structlog.testing.capture_logs() as logs:
        drop_attachment(
            markers,
            structlog.get_logger("t"),
            "telegram",
            name="x",
            content_type="application/pdf",
            size=1,
            reason="rejected",
            builder=broken,
        )
    assert markers == [FALLBACK_MARKER]
    events = [e["event"] for e in logs]
    assert "telegram.inbound.attachment_marker_failed" in events


def test_drop_attachment_builder_and_logger_failure_never_raise() -> None:
    class Bad:
        def warning(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("logger broken")

    def broken(*a: Any, **k: Any) -> str:
        raise RuntimeError("builder bug")

    markers: list[str] = []
    drop_attachment(
        markers, Bad(), "signal", name="x", content_type=None, size=None, reason="x", builder=broken
    )
    assert markers == [FALLBACK_MARKER]


def test_drop_attachment_stdlib_logger_reserved_key_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("t.drop")
    markers: list[str] = []
    with caplog.at_level(logging.WARNING, logger="t.drop"):
        drop_attachment(
            markers, logger, "signal", name="x.pdf", content_type="application/pdf",
            size=7441443, reason="over_cap",
        )
    [rec] = [r for r in caplog.records if r.name == "t.drop"]
    assert rec.__dict__["attachment_name"] == "x.pdf"
    assert markers and markers[0] != FALLBACK_MARKER
