"""Unit tests for the inbound-admission *operator surface* (#1501 — PR2).

The spine PR (#1500) fail-closed the runtime path; this PR gives the operator
the surface to set / read / default the policy safely. This suite pins, at the
wire + model + service layer (no DB):

* the ``InboundPolicyReplace`` request wrapper's Replace semantics — empty
  ``AllowList`` 422s, a partial ``allow_list`` missing ``chat_ids`` 422s,
  unknown / missing ``kind`` 422s, ``deny_all`` / ``allow_all`` accepted;
* ``effective_inbound_policy`` — the single home of the NULL → ``DenyAll``
  default — and its echo onto ``Connection.inbound_policy_effective``;
* the ``Connection`` echo per stored kind (NULL / allow_all / allow_list /
  deny_all), and that the echo is read-only (rejected on ``ConnectionCreate``);
* the ``set_inbound_policy`` / ``default_inbound_policy_if_unset`` query SQL
  shape (parametrized, NULL-guarded, never clobbers an operator-set policy).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from aios.models.connections import Connection, ConnectionCreate
from aios.models.inbound_policy import (
    AllowAll,
    AllowList,
    DenyAll,
    InboundPolicyReplace,
    effective_inbound_policy,
)

_REPLACE = TypeAdapter(InboundPolicyReplace)


# ─── InboundPolicyReplace: Replace (required-on-update) wire semantics ────────


def test_replace_accepts_deny_all() -> None:
    body = _REPLACE.validate_python({"kind": "deny_all"})
    assert isinstance(body.root, DenyAll)


def test_replace_accepts_allow_all() -> None:
    body = _REPLACE.validate_python({"kind": "allow_all"})
    assert isinstance(body.root, AllowAll)


def test_replace_accepts_allow_list() -> None:
    body = _REPLACE.validate_python({"kind": "allow_list", "chat_ids": ["a", "b"]})
    assert isinstance(body.root, AllowList)
    assert body.root.chat_ids == ["a", "b"]


def test_replace_empty_allow_list_422s() -> None:
    # Empty list is never a silent deny-all — it must fail at the write edge.
    with pytest.raises(ValidationError):
        _REPLACE.validate_python({"kind": "allow_list", "chat_ids": []})


def test_replace_partial_allow_list_missing_chat_ids_422s() -> None:
    # Required-on-update: a partial body can neither widen nor re-default.
    with pytest.raises(ValidationError):
        _REPLACE.validate_python({"kind": "allow_list"})


def test_replace_missing_kind_422s() -> None:
    with pytest.raises(ValidationError):
        _REPLACE.validate_python({"chat_ids": ["a"]})


def test_replace_unknown_kind_422s() -> None:
    # A deferred / unknown kind (e.g. the not-yet-built deny_list) must 422.
    with pytest.raises(ValidationError):
        _REPLACE.validate_python({"kind": "deny_list"})


def test_replace_extra_field_422s() -> None:
    # extra="forbid" on each member: an unexpected key 422s, not a silent drop.
    with pytest.raises(ValidationError):
        _REPLACE.validate_python({"kind": "deny_all", "bogus": 1})


def test_replace_wire_shape_is_bare_union() -> None:
    # RootModel dumps the bare {"kind": ...} shape — no envelope key.
    body = InboundPolicyReplace(root=AllowList(chat_ids=["x"]))
    assert body.model_dump() == {"kind": "allow_list", "chat_ids": ["x"]}


# ─── effective_inbound_policy: the single NULL → DenyAll default home ─────────


def test_effective_null_is_deny_all() -> None:
    eff = effective_inbound_policy(None)
    assert isinstance(eff, DenyAll)


def test_effective_passes_through_stored() -> None:
    stored = AllowList(chat_ids=["op"])
    assert effective_inbound_policy(stored) is stored


# ─── Connection echo: inbound_policy_effective per stored kind ────────────────


def _connection(*, inbound_policy: Any) -> Connection:
    now = datetime.now(UTC)
    return Connection(
        id="conn_1",
        connector="signal",
        external_account_id="+15551234567",
        metadata={},
        secrets_set=False,
        created_at=now,
        updated_at=now,
        inbound_policy=inbound_policy,
        inbound_policy_effective=effective_inbound_policy(inbound_policy),
    )


def test_echo_null_column_is_deny_all() -> None:
    conn = _connection(inbound_policy=None)
    assert isinstance(conn.inbound_policy_effective, DenyAll)


def test_echo_allow_all() -> None:
    conn = _connection(inbound_policy=AllowAll())
    assert isinstance(conn.inbound_policy_effective, AllowAll)


def test_echo_allow_list() -> None:
    conn = _connection(inbound_policy=AllowList(chat_ids=["c1"]))
    assert isinstance(conn.inbound_policy_effective, AllowList)
    assert conn.inbound_policy_effective.chat_ids == ["c1"]


def test_echo_deny_all() -> None:
    conn = _connection(inbound_policy=DenyAll())
    assert isinstance(conn.inbound_policy_effective, DenyAll)


def test_echo_default_when_omitted_is_deny_all() -> None:
    # Even constructed without the field, the read model defaults fail-closed.
    now = datetime.now(UTC)
    conn = Connection(
        id="c",
        connector="signal",
        external_account_id="+1",
        metadata={},
        secrets_set=False,
        created_at=now,
        updated_at=now,
    )
    assert isinstance(conn.inbound_policy_effective, DenyAll)


# ─── Echo is read-only on the write model ────────────────────────────────────


def test_create_rejects_inbound_policy_effective() -> None:
    # Server-derived; the write model (extra="forbid") rejects it as input.
    with pytest.raises(ValidationError):
        ConnectionCreate.model_validate(
            {
                "connector": "signal",
                "external_account_id": "acct",
                "inbound_policy_effective": {"kind": "allow_all"},
            }
        )


def test_create_rejects_inbound_policy_input() -> None:
    # The stored column is likewise server-internal, not a create input.
    with pytest.raises(ValidationError):
        ConnectionCreate.model_validate(
            {
                "connector": "signal",
                "external_account_id": "acct",
                "inbound_policy": {"kind": "allow_all"},
            }
        )


# ─── Query SQL shape: parametrized, NULL-guarded, archived-refusing ──────────


async def test_set_connection_inbound_policy_writes_jsonb_param() -> None:
    from aios.db.queries import connections as q

    now = datetime.now(UTC)
    fake_row = {
        "id": "conn_1",
        "connector": "signal",
        "external_account_id": "+1",
        "metadata": {},
        "secrets_ciphertext": None,
        "created_by_type": None,
        "created_by_ref": None,
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
        "inbound_policy": {"kind": "allow_list", "chat_ids": ["a"]},
        "binding_session_id": None,
        "binding_session_template_id": None,
        "binding_created_at": None,
    }
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=fake_row)

    result = await q.set_connection_inbound_policy(
        conn, "conn_1", policy=AllowList(chat_ids=["a"]), account_id="acc_1"
    )

    sql, *args = conn.fetchrow.await_args.args
    assert "UPDATE connections" in sql
    assert "inbound_policy = $2::jsonb" in sql
    assert "archived_at IS NULL" in sql
    # The serialized policy is passed as a bind param (no f-string injection).
    assert args[0] == "conn_1"
    assert '"allow_list"' in args[1]
    assert args[2] == "acc_1"
    assert isinstance(result.inbound_policy_effective, AllowList)


async def test_default_inbound_policy_if_unset_is_null_guarded() -> None:
    from aios.db.queries import connections as q

    conn = MagicMock()
    conn.execute = AsyncMock()
    await q.default_inbound_policy_if_unset(conn, "conn_1", account_id="acc_1")

    sql, *args = conn.execute.await_args.args
    assert "UPDATE connections" in sql
    assert "inbound_policy IS NULL" in sql  # never clobbers an operator-set policy
    assert '"deny_all"' in sql
    assert args == ["conn_1", "acc_1"]


# ─── default-closed at bind: attach / configure-per-chat call the defaulter ──


def _bind_pool() -> tuple[MagicMock, MagicMock]:
    """Pool whose acquire()/transaction() are async-context no-ops."""

    class _AsyncCm:
        def __init__(self, value: Any = None) -> None:
            self._value = value

        async def __aenter__(self) -> Any:
            return self._value

        async def __aexit__(self, *_a: Any) -> None:
            return None

    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_AsyncCm())
    conn.fetchrow = AsyncMock(return_value={"archived_at": None})
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCm(conn))
    return pool, conn


async def test_attach_defaults_inbound_policy_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.services import connections as svc

    conn_obj = _connection(inbound_policy=DenyAll())

    pool, conn = _bind_pool()
    session = MagicMock(archived_at=None, parent_run_id=None)
    monkeypatch.setattr(
        "aios.services.connections.queries.get_session_bare", AsyncMock(return_value=session)
    )
    monkeypatch.setattr("aios.services.connections.queries.insert_binding", AsyncMock())
    default_mock = AsyncMock()
    monkeypatch.setattr(
        "aios.services.connections.queries.default_inbound_policy_if_unset", default_mock
    )
    monkeypatch.setattr(
        "aios.services.connections.queries.get_connection", AsyncMock(return_value=conn_obj)
    )
    monkeypatch.setattr("aios.services.connections.queries.notify_connection_change", AsyncMock())
    monkeypatch.setattr("aios.services.connections._evict_sandbox_for_resource_change", MagicMock())

    await svc.attach_connection(pool, "conn_1", session_id="ses_1", account_id="acc_1")

    default_mock.assert_awaited_once_with(conn, "conn_1", account_id="acc_1")


async def test_configure_per_chat_defaults_inbound_policy_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aios.services import connections as svc

    conn_obj = _connection(inbound_policy=DenyAll())
    pool, conn = _bind_pool()
    # configure_per_chat now validates the template account-scoped inside the
    # tx (#1708); mock a live (un-archived) own-account template so the bind
    # proceeds to the policy default under test.
    template = MagicMock(archived_at=None)
    monkeypatch.setattr(
        "aios.services.connections.queries.get_session_template",
        AsyncMock(return_value=template),
    )
    monkeypatch.setattr("aios.services.connections.queries.insert_binding", AsyncMock())
    default_mock = AsyncMock()
    monkeypatch.setattr(
        "aios.services.connections.queries.default_inbound_policy_if_unset", default_mock
    )
    monkeypatch.setattr(
        "aios.services.connections.queries.get_connection", AsyncMock(return_value=conn_obj)
    )

    await svc.configure_per_chat(pool, "conn_1", session_template_id="stpl_1", account_id="acc_1")

    default_mock.assert_awaited_once_with(conn, "conn_1", account_id="acc_1")
