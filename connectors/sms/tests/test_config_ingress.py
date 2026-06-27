"""Ingress config parsing (allowedHosts / trusted-proxies / self-test)."""

from __future__ import annotations

from aios_sms.config import Settings


def test_allowed_hosts_parses_comma_separated_env(monkeypatch) -> None:
    monkeypatch.setenv("AIOS_SMS_ALLOWED_HOSTS", "sms.example.com, sms2.example.com")
    s = Settings()
    assert s.allowed_hosts == frozenset({"sms.example.com", "sms2.example.com"})


def test_trusted_proxies_parses_space_separated_env(monkeypatch) -> None:
    monkeypatch.setenv("AIOS_SMS_TRUSTED_PROXIES", "10.0.0.0/8 192.0.2.7")
    s = Settings()
    assert s.trusted_proxies == frozenset({"10.0.0.0/8", "192.0.2.7"})


def test_ingress_sets_default_empty_fail_closed() -> None:
    s = Settings()
    # Default = empty sets ⇒ forwarded-header fallback disabled (fail closed).
    assert s.allowed_hosts == frozenset()
    assert s.trusted_proxies == frozenset()


def test_public_port_defaults_to_443() -> None:
    assert Settings().public_port == 443


def test_self_test_defaults_enabled_and_fail_fast() -> None:
    s = Settings()
    assert s.self_test_enabled is True
    assert s.self_test_fail_fast is True


def test_self_test_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AIOS_SMS_SELF_TEST_ENABLED", "false")
    monkeypatch.setenv("AIOS_SMS_SELF_TEST_FAIL_FAST", "false")
    s = Settings()
    assert s.self_test_enabled is False
    assert s.self_test_fail_fast is False
