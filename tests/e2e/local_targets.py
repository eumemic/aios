"""Narrowly-scoped opt-in for E2E fixtures that legitimately need a local target.

Background (PR #1931 review). ``tests/e2e/__init__.py`` used to set
``AIOS_OAUTH_ALLOW_INSECURE_HOSTS=127.0.0.1,localhost`` at package-import time.
Because that is a *process-global* env write, it allowlisted loopback for every
test in the E2E process and thereby disarmed both enforcement points of the
#861 policy at once:

* the write boundary — ``validate_outbound_target_url`` returns early for an
  allowlisted host, so a loopback ``mcp_servers[].url`` / vault ``target_url``
  could be persisted;
* the connection boundary — ``PinnedTransport`` skips pinning entirely for an
  allowlisted host, so the loopback address could actually be dialed.

The casualty was ``test_vault_oauth.py::test_rejects_insecure_target``, whose
``https://localhost/mcp`` and ``https://127.0.0.1/mcp`` cases are supposed to
raise ``OAuthFlowError``: with loopback allowlisted, ``_guard_url`` returned
early and the test only still passed by way of the unrelated cleartext-http
case. A green suite proved nothing about the loopback path.

This module replaces that with an explicit, *scoped* opt-in. Use it only where
the test's subject genuinely is a local service (an in-process uvicorn app, a
loopback recorder upstream), never as a blanket relaxation:

    from tests.e2e.local_targets import allow_local_targets

    with allow_local_targets("127.0.0.1:8080"):
        ...                     # policy relaxed for this host, here only

or the ``local_targets_allowed`` fixture for a fixture/test-scoped window.
Outside those windows the default policy stays ARMED, so the loopback
regression tests keep failing closed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import pytest

# The loopback spellings a local test service is reachable under. Kept here (not
# inlined at call sites) so the opt-in surface is one reviewable list.
LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")


@contextmanager
def allow_local_targets(*hosts: str) -> Iterator[frozenset[str]]:
    """Allowlist ``hosts`` (default: loopback) for the duration of the block.

    Patches ``AIOS_OAUTH_ALLOW_INSECURE_HOSTS`` — the SINGLE operator allowlist
    that the write-boundary validator, ``PinnedTransport``, and the
    cleartext-credential guard all consult, so relaxing it here cannot make the
    two boundaries disagree — and clears the ``get_settings`` cache on both
    edges so the window opens and closes deterministically.

    Entries may be bare hosts or ``host:port``. Existing entries are preserved
    (unioned), so nesting this inside a test that already allowlists a mock
    provider host does not silently drop that host.
    """
    from aios.config import get_settings

    requested = frozenset(hosts) if hosts else frozenset(LOOPBACK_HOSTS)
    existing = frozenset(
        h.strip() for h in os.getenv("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", "").split(",") if h.strip()
    )
    merged = existing | requested
    with mock.patch.dict(os.environ, {"AIOS_OAUTH_ALLOW_INSECURE_HOSTS": ",".join(sorted(merged))}):
        get_settings.cache_clear()
        try:
            yield merged
        finally:
            get_settings.cache_clear()


@pytest.fixture
def local_targets_allowed() -> Iterator[frozenset[str]]:
    """Fixture form of :func:`allow_local_targets` for loopback hosts.

    Request it ONLY from a test whose subject is a real local service. It is
    deliberately not autouse and not session-scoped: the default policy must
    stay armed for every test that does not name it.
    """
    with allow_local_targets() as hosts:
        yield hosts
