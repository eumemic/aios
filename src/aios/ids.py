"""Prefixed Crockford-base32 ULIDs.

Every aios resource id has the form ``<prefix>_<26-char-ULID>``. ULIDs are
time-ordered, so DB indexes on append-heavy tables (the events table in
particular) stay tight. The prefix makes ids self-describing in logs and
error messages without an extra type lookup.

Example::

    >>> make_id("agent")
    'agent_01HQR2K7VXBZ9MNPL3WYCT8F'
"""

from __future__ import annotations

from typing import Final

from ulid import ULID

# Canonical prefixes for every resource kind. Centralized so callers don't
# accidentally use a typo'd prefix that the DB would happily accept.
AGENT: Final = "agent"
ENVIRONMENT: Final = "env"
SESSION: Final = "sess"
EVENT: Final = "evt"
CREDENTIAL: Final = "cred"
VAULT: Final = "vlt"
VAULT_CREDENTIAL: Final = "vcr"
SKILL: Final = "skl"
CONNECTION: Final = "conn"
SESSION_TEMPLATE: Final = "stpl"
# Connector subsystem (#328 PR 2+). The aios_connectors module uses
# these prefixes to mint ids via ``make_id``; the PR 2 migration
# also backfills binding ids prefixed with ``bnd_`` (but with a
# random-hex body, not a ULID) — backfilled rows are valid ``text``
# PKs that won't survive ``split_id`` parsing.  No current call site
# does that on a binding id; if one is added later, generate ULIDs
# in the migration or relax ``split_id``.
BINDING: Final = "bnd"
RUNTIME_TOKEN: Final = "rtk"
MEMORY_STORE: Final = "memstore"
MEMORY: Final = "mem"
MEMORY_VERSION: Final = "memver"
GITHUB_REPOSITORY: Final = "ghrepo"
FILE: Final = "file"
MANAGEMENT_CALL: Final = "mgmt"
ACCOUNT: Final = "acc"
ACCOUNT_KEY: Final = "acckey"

_PREFIXES: Final = frozenset(
    {
        AGENT,
        ENVIRONMENT,
        SESSION,
        EVENT,
        CREDENTIAL,
        VAULT,
        VAULT_CREDENTIAL,
        SKILL,
        CONNECTION,
        SESSION_TEMPLATE,
        MEMORY_STORE,
        MEMORY,
        MEMORY_VERSION,
        GITHUB_REPOSITORY,
        FILE,
        BINDING,
        RUNTIME_TOKEN,
        MANAGEMENT_CALL,
        ACCOUNT,
        ACCOUNT_KEY,
    }
)


def make_id(prefix: str) -> str:
    """Generate a fresh prefixed ULID id for ``prefix``.

    Raises ``ValueError`` if ``prefix`` isn't one of the canonical aios
    resource prefixes — this catches typos before they reach the DB.
    """
    if prefix not in _PREFIXES:
        raise ValueError(f"unknown id prefix {prefix!r}; expected one of {sorted(_PREFIXES)}")
    return f"{prefix}_{ULID()}"


def split_id(value: str) -> tuple[str, str]:
    """Split a prefixed id into ``(prefix, ulid)``.

    Useful for assertions and logging. Raises ``ValueError`` for malformed ids.
    """
    prefix, sep, ulid_part = value.partition("_")
    if not sep or prefix not in _PREFIXES or len(ulid_part) != 26:
        raise ValueError(f"malformed prefixed id: {value!r}")
    return prefix, ulid_part
