"""Build-time foreclosure for the credential-free host's env boundary.

``host_launcher._CHILD_ENV_ALLOWLIST`` is the *entire* "the child subprocess
never inherits a secret" guarantee (CLAUDE.md §Workflows: the master
``CryptoBox`` and the all-accounts pool are never inherited). The child IS the
isolation boundary — author code is assumed able to read the child's whole
environment (``gate.__globals__['os'].environ``), so a single secret-bearing var
in the allowlist breaches every tenant's credentials.

The old guard was a hand-curated allowlist plus a leak test that hardcoded three
sentinel names. Because the allowlist is deny-by-default, those three were
*trivially* excluded — that test passed vacuously and constrained nothing a
future dev might **add**. Here we instead assert the allowlist is **disjoint**
from a secret-shaped namespace that is **derived** (not hand-listed):

* every ``Settings`` field's env name — ``AIOS_`` + field name, uppercased,
  matching the pydantic-settings ``env_prefix="AIOS_"`` + ``case_sensitive=False``
  contract. ``Settings.model_fields`` is the single source of truth for every
  in-house env var, so a newly-added setting joins the exclusion set with no
  test edit; and
* the provider-key / generic-secret shapes: any name ending ``_API_KEY``,
  ``_KEY``, ``_TOKEN``, or ``_SECRET``.

So an unaware dev who adds ``AIOS_FOO`` (a new setting) or any ``*_API_KEY`` to
the allowlist to "let the child launch" fails CI at the moment of addition. This
is a build-time assertion, not a runtime validator that ships-when-wrong, and
not a "never widen" comment a reviewer must remember to honor.
"""

from __future__ import annotations

from aios.config import Settings
from aios.workflows.host_launcher import _CHILD_ENV_ALLOWLIST

# Suffixes that mark a value as secret-shaped regardless of source: provider API
# keys (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / …) and the generic
# credential conventions. ``_API_KEY`` is subsumed by ``_KEY`` but is spelled out
# to document the provider-key case the issue calls out explicitly.
_SECRET_NAME_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_KEY", "_TOKEN", "_SECRET")


def _settings_env_names() -> frozenset[str]:
    """The env var name pydantic-settings reads for each ``Settings`` field.

    ``env_prefix="AIOS_"`` + ``case_sensitive=False`` means the field ``vault_key``
    is populated from ``AIOS_VAULT_KEY``. Deriving from ``model_fields`` makes the
    in-house secret set track the model: add a setting, get the exclusion for free.
    """
    prefix = Settings.model_config.get("env_prefix", "")
    return frozenset((prefix + name).upper() for name in Settings.model_fields)


def _derived_secret_namespace() -> frozenset[str]:
    """The names the allowlist must never admit: every in-house ``AIOS_*`` setting
    plus every provider-key / generic-secret-shaped name in the allowlist itself."""
    suffix_shaped = frozenset(
        name for name in _CHILD_ENV_ALLOWLIST if name.upper().endswith(_SECRET_NAME_SUFFIXES)
    )
    return _settings_env_names() | suffix_shaped


def test_settings_env_names_cover_known_in_house_secrets() -> None:
    """Sanity-check the derivation: the canonical in-house secrets the old vacuous
    test hardcoded are all present in the derived set (so the disjointness assert
    below is not vacuous itself)."""
    env_names = _settings_env_names()
    for known_secret in ("AIOS_VAULT_KEY", "AIOS_EGRESS_CA_KEY", "AIOS_BOOTSTRAP_TOKEN"):
        assert known_secret in env_names, known_secret


def test_child_env_allowlist_is_disjoint_from_secret_namespace() -> None:
    """The load-bearing assertion: no launch-essential the child inherits is a
    secret. Derived from ``Settings.model_fields`` + provider-key shapes, so a
    future dev who adds an ``AIOS_*`` setting name (or any ``*_API_KEY`` / ``*_KEY``
    / ``*_TOKEN`` / ``*_SECRET``) to the allowlist trips this at build time."""
    leaked = _CHILD_ENV_ALLOWLIST & _derived_secret_namespace()
    assert leaked == frozenset(), (
        "_CHILD_ENV_ALLOWLIST admits secret-shaped env var(s) "
        f"{sorted(leaked)} — the child subprocess would inherit them, breaching the "
        "credential-free guarantee. Remove them; never widen the allowlist toward secrets."
    )


def test_disjointness_assert_catches_an_added_secret_var() -> None:
    """Guards the guard: prove a future addition would actually be caught, so this
    test never silently goes vacuous the way the 3-name leak test did. Both an
    ``AIOS_*`` setting name and an arbitrary provider-style key must be detected."""
    secret_ns = _derived_secret_namespace()
    for hypothetical in ("AIOS_VAULT_KEY", "AIOS_DB_URL", "OPENAI_API_KEY", "STRIPE_SECRET"):
        widened = _CHILD_ENV_ALLOWLIST | {hypothetical}
        # the suffix-shaped (``OPENAI_API_KEY`` / ``STRIPE_SECRET``) names enter via the
        # allowlist-scan branch, so recompute the namespace against the widened set.
        suffix_shaped = frozenset(n for n in widened if n.upper().endswith(_SECRET_NAME_SUFFIXES))
        widened_secret_ns = secret_ns | suffix_shaped
        assert widened & widened_secret_ns, hypothetical
