"""Drift guard for the inbound-admission design of record (epic #1499).

The epic names ``docs/design/inbound-admission.md`` as the durable home of the
settled design. This guard pins that the doc exists and that its load-bearing
claims track the code rather than silently re-rotting:

* every ``InboundPolicy`` union ``kind`` is documented (a new admission kind
  must be written up, per the kind-not-flag growth rule);
* every admission-relevant ``InboundDrop`` reason is documented with the
  non-fatal status the router actually maps it to, and that status is
  non-fatal for the connector-http runner (``_is_fatal_inbound_status``);
* the NULL → ``DenyAll`` fail-closed default the doc states is what
  ``effective_inbound_policy`` does.

Read-only over source-of-truth files; no network / DB / Docker.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from aios.api.routers.connectors import _inbound_drop_error
from aios.models.inbound_policy import DenyAll, InboundPolicy, effective_inbound_policy
from aios.services.inbound import InboundDrop
from aios_connector_http.runner import _is_fatal_inbound_status

DOC = Path(__file__).resolve().parents[2] / "docs/design/inbound-admission.md"

_ADMISSION_DROPS = (
    InboundDrop.DENIED_BY_POLICY,
    InboundDrop.PENDING_APPROVAL,
    InboundDrop.RATE_LIMITED,
)


def _policy_kinds() -> list[str]:
    union = typing.get_args(InboundPolicy)[0]
    return [member.model_fields["kind"].default for member in typing.get_args(union)]


def _doc() -> str:
    assert DOC.is_file(), f"{DOC} is missing — epic #1499 names it as the design of record"
    return DOC.read_text(encoding="utf-8")


def test_policy_kinds_enumeration_is_nonempty() -> None:
    assert set(_policy_kinds()) >= {"allow_all", "allow_list", "deny_all"}


@pytest.mark.parametrize("kind", _policy_kinds())
def test_doc_documents_every_policy_kind(kind: str) -> None:
    assert f"`{kind}`" in _doc(), f"inbound-admission.md does not document kind `{kind}`"


@pytest.mark.parametrize("drop", _ADMISSION_DROPS)
def test_doc_documents_admission_drops_with_their_non_fatal_status(drop: InboundDrop) -> None:
    status = _inbound_drop_error(drop).status_code
    assert _is_fatal_inbound_status(status) is False
    assert f"| `{drop.value}` | {status} |" in _doc(), (
        f"inbound-admission.md must list `{drop.value}` → {status} in its drop table"
    )


def test_doc_states_null_resolves_to_deny_all() -> None:
    assert effective_inbound_policy(None) == DenyAll()
    assert "NULL" in _doc() and "`deny_all`" in _doc()
