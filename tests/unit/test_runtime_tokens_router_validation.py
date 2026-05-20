"""Unit tests for ``RuntimeTokenIssue`` wire validation — connection_ids scope (#350).

The ``connection_ids`` field is the optional allowlist scope: when set
on issue, the resulting runtime token can only see/operate on the
listed connection IDs. When omitted, the token behaves as before
(``None`` => unscoped, sees all connections of its connector type).
"""

from __future__ import annotations

from aios.models.runtime_tokens import RuntimeTokenIssue


class TestRuntimeTokenIssueConnectionIds:
    def test_accepts_connection_ids_allowlist(self) -> None:
        body = RuntimeTokenIssue.model_validate(
            {"connector": "echo", "connection_ids": ["c1", "c2"]}
        )
        assert body.connection_ids == ["c1", "c2"]

    def test_defaults_to_none_when_omitted(self) -> None:
        body = RuntimeTokenIssue.model_validate({"connector": "echo"})
        assert body.connection_ids is None

    def test_accepts_empty_list(self) -> None:
        """``[]`` is a valid empty allowlist (zero connections accessible)
        — distinct from ``None`` (unscoped). Don't coerce."""
        body = RuntimeTokenIssue.model_validate({"connector": "echo", "connection_ids": []})
        assert body.connection_ids == []
