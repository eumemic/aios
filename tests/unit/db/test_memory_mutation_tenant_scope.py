"""Tenant-scope tripwires for nested memory mutations.

These repository functions are public query primitives.  They must be safe when
called directly, rather than relying on a service-layer point read to authorize
the same identifiers first.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from aios.db.queries import memory_stores


def _normalized_source(function: Callable[..., Any]) -> str:
    return " ".join(inspect.getsource(function).split())


def test_update_memory_scopes_lock_and_write_in_sql() -> None:
    source = _normalized_source(memory_stores.update_memory_with_version)

    assert "AND deleted_at IS NULL AND account_id = $3 FOR UPDATE" in source
    assert "WHERE memory_store_id = $6 AND id = $7 AND account_id = $8 RETURNING *" in source


def test_delete_memory_scopes_lock_and_write_in_sql() -> None:
    source = _normalized_source(memory_stores.delete_memory_with_version)

    assert "AND deleted_at IS NULL AND account_id = $3 FOR UPDATE" in source
    assert "WHERE memory_store_id = $1 AND id = $2 AND account_id = $3" in source


def test_redact_memory_version_scopes_write_in_sql() -> None:
    source = _normalized_source(memory_stores.redact_memory_version)

    assert "WHERE memory_store_id = $3 AND id = $4 AND account_id = $5 RETURNING *" in source
