"""Regression coverage for issue #1682's transport-failure translation.

``run_or_die`` is the CLI's outermost handler. The generated SDK ops
(``call_single``/``render_paginated``) raise raw ``httpx`` errors on a
down/unreachable server, so this shim is the only layer that can translate
them into a clean ``typer.Exit(1)`` instead of a leaked ``httpx`` traceback.

The original implementation caught only ``httpx.ConnectError`` and
``httpx.TimeoutException``. The other transport-failure leaves of httpx's
``RequestError`` tree (``ReadError``/``WriteError``/``ProxyError``/
``*ProtocolError``/...) are SIBLINGS of those two, not subclasses, so they
bypassed both ``except`` blocks and printed a raw ``httpx`` traceback (the
CLI runs typer with ``pretty_exceptions_enable=False``). These tests pin
the widened ``httpx.RequestError`` catch so the leak can't regress.
"""

from __future__ import annotations

import httpx
import pytest
import typer

from aios.cli.runtime import run_or_die


def test_run_or_die_translates_connect_error(capsys: pytest.CaptureFixture[str]) -> None:
    def fn() -> int | None:
        raise httpx.ConnectError("refused")

    with pytest.raises(typer.Exit) as excinfo:
        run_or_die(fn)

    assert excinfo.value.exit_code == 1
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)
    err = capsys.readouterr().err
    assert "connection_error" in err
    assert "could not connect" in err


def test_run_or_die_translates_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    def fn() -> int | None:
        raise httpx.ReadTimeout("timed out reading body")

    with pytest.raises(typer.Exit) as excinfo:
        run_or_die(fn)

    assert excinfo.value.exit_code == 1
    assert isinstance(excinfo.value.__cause__, httpx.TimeoutException)
    err = capsys.readouterr().err
    assert "timeout" in err


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(httpx.ReadError("connection reset mid-body"), id="ReadError"),
        pytest.param(httpx.WriteError("failed writing request body"), id="WriteError"),
        pytest.param(httpx.CloseError("connection closed"), id="CloseError"),
        pytest.param(httpx.ProxyError("proxy handshake failed"), id="ProxyError"),
        pytest.param(httpx.LocalProtocolError("bad local framing"), id="LocalProtocolError"),
        pytest.param(httpx.RemoteProtocolError("bad remote framing"), id="RemoteProtocolError"),
        pytest.param(httpx.UnsupportedProtocol("no protocol support"), id="UnsupportedProtocol"),
        pytest.param(httpx.DecodingError("could not decode body"), id="DecodingError"),
        pytest.param(httpx.TooManyRedirects("too many redirects"), id="TooManyRedirects"),
    ],
)
def test_run_or_die_translates_request_error_leaves(
    exc: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every ``RequestError`` leaf is translated to ``typer.Exit(1)`` with a
    clean ``connection_error:`` message — never a leaked httpx traceback."""

    def fn() -> int | None:
        raise exc

    with pytest.raises(typer.Exit) as excinfo:
        run_or_die(fn)

    assert excinfo.value.exit_code == 1
    assert isinstance(excinfo.value.__cause__, httpx.RequestError)
    err = capsys.readouterr().err
    assert "connection_error" in err
    assert "request failed" in err


def test_run_or_die_does_not_swallow_non_transport_exceptions() -> None:
    """The httpx widening must stay scoped to httpx: a non-transport,
    non-API exception still bubbles unchanged (issue #1682)."""

    def fn() -> int | None:
        raise ValueError("not a transport error")

    with pytest.raises(ValueError, match="not a transport error"):
        run_or_die(fn)
