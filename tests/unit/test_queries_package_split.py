"""Structural guards for the ``aios.db.queries`` per-resource package split (#854).

The 8928-line god module ``db/queries/__init__.py`` was split into 13
per-domain sibling modules, re-exported from ``__init__`` so every call site
stays byte-identical. These tests pin the split contract:

* each query function physically lives in its expected submodule
  (``.__module__``);
* the package root re-exports the SAME object the submodule defines
  (identity, so ``patch.object(queries, "X")`` reaches the real callee);
* underscore-prefixed helpers, classes, NamedTuples, and constants are
  re-exported too (not just public ``async def``\\s);
* the six tests-patch-this functions are called through the package
  attribute by their internal callers, so patching keeps working.

Nothing here touches Postgres — it is pure import/introspection.
"""

from __future__ import annotations

import inspect

import aios.db.queries as q
from aios.db.queries import (
    accounts,
    agents,
    connections,
    environments,
    events,
    files,
    management_calls,
    memory_stores,
    sandboxes,
    session_templates,
    sessions,
    skills,
    triggers,
    vaults,
)


def test_functions_live_in_expected_submodules() -> None:
    assert environments.get_environment.__module__ == "aios.db.queries.environments"
    assert agents.get_agent.__module__ == "aios.db.queries.agents"
    assert sessions.get_session.__module__ == "aios.db.queries.sessions"
    assert events.append_event.__module__ == "aios.db.queries.events"
    assert vaults.resolve_vault_credential.__module__ == "aios.db.queries.vaults"
    assert vaults.update_vault_credential.__module__ == "aios.db.queries.vaults"
    assert skills.resolve_skill_refs.__module__ == "aios.db.queries.skills"
    assert connections.get_active_binding.__module__ == "aios.db.queries.connections"
    assert connections.list_recent_chat_ids.__module__ == "aios.db.queries.connections"
    assert connections.reparent_connection.__module__ == "aios.db.queries.connections"
    assert session_templates.get_session_template.__module__ == "aios.db.queries.session_templates"
    assert memory_stores.insert_memory_with_version.__module__ == "aios.db.queries.memory_stores"
    assert files.insert_file.__module__ == "aios.db.queries.files"
    for function in (
        management_calls.insert_management_call,
        management_calls.list_pending_management_calls_for_connector,
        management_calls.get_management_call,
        management_calls.mark_management_call_resolved,
        management_calls.notify_management_call_dispatch,
        management_calls.notify_management_call_result,
    ):
        assert function.__module__ == "aios.db.queries.management_calls"
    assert accounts.bootstrap_root_account.__module__ == "aios.db.queries.accounts"
    assert sandboxes.unscoped_set_session_snapshot.__module__ == "aios.db.queries.sandboxes"
    assert triggers.delete_trigger_by_id.__module__ == "aios.db.queries.triggers"


def test_reexport_identity() -> None:
    assert q.append_event is events.append_event
    assert q.get_session is sessions.get_session
    assert q.get_session_template is session_templates.get_session_template
    assert q.get_active_binding is connections.get_active_binding
    for name in (
        "insert_management_call",
        "list_pending_management_calls_for_connector",
        "get_management_call",
        "mark_management_call_resolved",
        "notify_management_call_dispatch",
        "notify_management_call_result",
    ):
        assert getattr(q, name) is getattr(management_calls, name)
    assert q.delete_trigger_by_id is triggers.delete_trigger_by_id
    assert q.update_vault_credential is vaults.update_vault_credential
    assert q.reparent_connection is connections.reparent_connection


def test_underscore_and_nonfunc_names_reexported() -> None:
    names = (
        "_SESSION_STATUS_EXPR",
        "_get_scoped",
        "_list_scoped",
        "_archive_scoped",
        "_build_set_assignments",
        "_escape_like",
        "_derive_is_error",
        "_derive_sender_name",
        "_clear_model_token_ratio_cache",
        "reparent_connection",
        "get_session_bare",
        "get_session_vault_ids",
        "get_environment",
        "read_events",
        "EnvVarCredentialEcho",
        "EnvVarCredentialRow",
        "ActiveBinding",
        "TriggerRow",
        "TriggerFireRef",
    )
    for name in names:
        assert hasattr(q, name), f"package root is missing re-export {name!r}"


def test_parse_jsonb_shim_is_deleted() -> None:
    # The jsonb codec (aios.db.pool.register_jsonb_codec) is the single source
    # of truth for JSONB decoding; the old passthrough shim must not return.
    import aios.db.queries as q

    assert not hasattr(q, "parse_jsonb")
    assert "parse_jsonb" not in q.__all__


def test_internal_callers_route_patched_fns_through_package() -> None:
    write_response_src = inspect.getsource(sessions.write_response_if_absent)
    assert "queries.append_event(" in write_response_src

    update_session_src = inspect.getsource(sessions.update_session)
    assert "queries.get_session(" in update_session_src

    update_template_src = inspect.getsource(session_templates.update_session_template)
    assert "queries.get_session_template(" in update_template_src

    # ``read_windowed_events`` (events.py) internally calls two functions that
    # tests patch at the package attribute — ``read_windowed_context_events``
    # (fallback load-all path) and ``model_token_class_ratios``
    # (test_windowed_ratio.py).  Both live in events.py too, so the same-module
    # calls must route through the package or the monkeypatch is bypassed. (The
    # retained-window range scan calls ``read_windowed_context_events`` bare on
    # purpose, so the fallback stub does not intercept it — that bare call is not
    # asserted here.)  Since #1609 the per-content-class calibration replaced the
    # removed scalar reader; the windowing path now routes through
    # ``model_token_class_ratios`` (which the windowed-ratio tests patch).
    windowed_src = inspect.getsource(events.read_windowed_events)
    assert "queries.read_windowed_context_events(" in windowed_src
    assert "queries.model_token_class_ratios(" in windowed_src
