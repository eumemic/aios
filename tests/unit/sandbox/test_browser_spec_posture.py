"""Posture pins for ``build_spec_from_browser`` (jarbot#106).

Three docstrings in the browser stack lean on claims about the browser
container's spec that no test asserted directly — most load-bearing:
``environment={}``, which is what makes the driver's
``AIOS_BROWSER_DRIVER_ALLOW_PRIVATE_NAV`` test knob (an SSRF-guard bypass)
unsettable through aios. If a future change threads env into the browser
spec, this fails instead of silently arming the knob path in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.network import BROWSER_NETWORK_NAME
from aios.sandbox.spec import BrowserImageUnconfiguredError, build_spec_from_browser


@pytest.fixture
def browser_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    monkeypatch.setattr(settings, "sandbox_browser_image", "aios-browser:test")


def test_browser_spec_posture(browser_settings: None) -> None:
    spec = build_spec_from_browser("acc_TESTOWNER")
    # No env injection — the driver's private-nav test knob (and any future
    # env-keyed behavior) is unreachable through aios by construction.
    assert spec.environment == {}
    # No route to the worker; the worker reaches the container via exec only.
    assert spec.host_gateway_alias is None
    # The plane is the ONLY mount.
    assert spec.extra_mounts == ()
    assert spec.workspace.sandbox_path == "/workspace"
    assert spec.network_name == BROWSER_NETWORK_NAME
    assert spec.seccomp_profile == get_settings().sandbox_browser_seccomp_profile


def test_browser_spec_rejects_non_account_owner(browser_settings: None) -> None:
    with pytest.raises(ValueError, match="account id"):
        build_spec_from_browser("ses_NOTANACCOUNT")


def test_browser_spec_requires_configured_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    monkeypatch.setattr(settings, "sandbox_browser_image", "")
    with pytest.raises(BrowserImageUnconfiguredError):
        build_spec_from_browser("acc_TESTOWNER")
