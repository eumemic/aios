"""Structural drift-guard for the env-var credential ``FROM/WHERE`` predicate (#1075).

The membership/resolution predicate that scopes which ``environment_variable``
credentials a session/run can read is the *cross-tenant credential-isolation
guarantee*: the double ``account_id = $2`` tenant scoping (binding row *and*
credential row), the ``archived_at IS NULL`` revocation filter, and the
``DISTINCT ON (secret_name)`` first-vault-wins ordering. It is embedded by three
query sites across two owners:

* ``list_session_env_var_credentials`` — provision-time set,
* ``list_session_env_var_credential_echoes`` — per-step drift echo (#877),
* ``list_run_env_var_credentials`` — workflow-run set (#882/#1011).

Before #1075 the run query inlined a *byte-identical copy* of the session body —
the exact "can NEVER silently diverge" failure the author's shared const exists
to prevent, reproduced one function away. Now both owner constants are derived
from a single ``_ENV_VAR_CREDENTIALS_FROM_WHERE`` template, so a future
scoping/revocation edit lands on session and run together by construction.

This test pins that single-sourcing structurally: the rendered session and run
bodies must be IDENTICAL after substituting back the owner tokens — i.e. they may
differ ONLY in ``session_vaults``/``sv``/``session_id`` vs
``wf_run_vaults``/``rv``/``run_id``. Modeled on
``test_terminal_status_vocabulary_single_sourced``. Pure introspection — no
Postgres.
"""

from __future__ import annotations

from aios.db.queries import vaults


def _normalize(body: str, table: str, alias: str, owner_col: str) -> str:
    """Replace this owner's table/alias/owner-column with neutral tokens so two
    owners' bodies are comparable iff they are structurally identical."""
    return body.replace(table, "<TABLE>").replace(owner_col, "<OWNER_COL>").replace(alias, "<A>")


def test_env_var_predicate_single_sourced_across_owners() -> None:
    """Session and run env-var credential bodies are the SAME template — they
    differ ONLY in owner table/alias/column, so the cross-tenant credential
    predicate can't drift between the session and run execution paths."""
    session_normalized = _normalize(
        vaults._SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE,
        table="session_vaults",
        alias="sv",
        owner_col="session_id",
    )
    run_normalized = _normalize(
        vaults._RUN_ENV_VAR_CREDENTIALS_FROM_WHERE,
        table="wf_run_vaults",
        alias="rv",
        owner_col="run_id",
    )
    assert session_normalized == run_normalized


def test_owner_constants_derive_from_one_template() -> None:
    """Both owner constants must be ``.format()`` renderings of the single
    ``_ENV_VAR_CREDENTIALS_FROM_WHERE`` template (the one home of the
    predicate), not hand-written twins."""
    assert (
        vaults._ENV_VAR_CREDENTIALS_FROM_WHERE.format(
            table="session_vaults", a="sv", owner_col="session_id"
        )
    ) == vaults._SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE
    assert (
        vaults._ENV_VAR_CREDENTIALS_FROM_WHERE.format(
            table="wf_run_vaults", a="rv", owner_col="run_id"
        )
    ) == vaults._RUN_ENV_VAR_CREDENTIALS_FROM_WHERE


def test_security_predicate_present_in_template() -> None:
    """The security-load-bearing clauses must live in the single template, so a
    scoping/revocation edit there reaches every owner. Guards against someone
    'simplifying' the template and dropping a tenant scope or the archival
    filter."""
    template = vaults._ENV_VAR_CREDENTIALS_FROM_WHERE
    # Double cross-tenant scope: the binding row AND the credential row.
    assert "{a}.account_id = $2" in template
    assert "vc.account_id = $2" in template
    # Revocation filter.
    assert "vc.archived_at IS NULL" in template
    # Owner scope and first-vault-wins ordering.
    assert "{a}.{owner_col} = $1" in template
    assert "ORDER BY vc.secret_name, {a}.rank" in template
    assert "vc.auth_type = 'environment_variable'" in template
