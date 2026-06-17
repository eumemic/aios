"""Unit tests for the IaC-sync v1 agent reconciler (issue #1282).

The script's load-bearing logic is the pure pipeline — load + validate manifests,
diff against the live read view (normalised so nested default-population differences
don't produce phantom drift), and decide create/update/noop — plus the fail-loud
guards: the duplicate-name guard, the create rewire-warning, the `--check` no-write
invariant, and the 409 version-mismatch. These tests exercise that logic OFFLINE
(no network, no live agent), plus the committed-manifest validation + no-secret
invariant, and the deploy-on-merge Action's drift guards.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

from aios.models.agents import AgentCreate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "reconcile_agents.py"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "reconcile-agents.yml"
_MANIFEST_DIR = _REPO_ROOT / "infra" / "agents"


def _load_script_module() -> ModuleType:
    # Mirror test_reregister_workflows.py's importlib harness: register before exec
    # so dataclasses/annotations resolve via sys.modules[cls.__module__].
    spec = importlib.util.spec_from_file_location("reconcile_agents", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ra = _load_script_module()


def _manifest(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "dev-implement",
        "model": "anthropic/claude-opus-4-8",
        "system": "do the thing",
        "tools": [{"type": "bash"}, {"type": "read"}],
        "http_servers": [],
        "litellm_extra": {},
        "window_min": 50000,
        "window_max": 150000,
    }
    base.update(over)
    return base


def _live_from(manifest: dict[str, Any], **over: Any) -> dict[str, Any]:
    """Build a plausible LIVE read view from a manifest: normalise the manifest's
    versioned fields through AgentCreate (default-populated, field-reordered) and
    bolt on the server-only fields a real GET returns."""
    canonical = AgentCreate(**manifest).model_dump()
    live: dict[str, Any] = {
        "id": "agent_01EXISTING",
        "version": 4,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "archived_at": None,
        **canonical,
    }
    live.update(over)
    return live


# ─── committed manifests: validate + no-secret invariant ─────────────────────


def _committed_manifest_paths() -> list[Path]:
    return sorted(_MANIFEST_DIR.glob("*.json"))


def test_manifest_dir_has_at_least_one_manifest() -> None:
    assert _committed_manifest_paths(), "expected at least one infra/agents/*.json manifest"


def test_every_committed_manifest_validates_against_agentcreate() -> None:
    """Extra-field / malformed manifests must fail HERE (AgentCreate is extra=forbid),
    not in prod. load_manifests is the production path; assert it accepts the dir."""
    manifests = ra.load_manifests(_MANIFEST_DIR)
    assert len(manifests) == len(_committed_manifest_paths())
    for m in manifests:
        AgentCreate(**m)  # redundant belt-and-braces; raises on any drift


def test_committed_manifests_carry_no_secret_shaped_literal() -> None:
    """No-secret invariant: a manifest may reference a credential by NAME but must
    never embed key material. Assert no key-shaped literal anywhere in the JSON, and
    no top-level api_key/token/secret key."""
    key_shaped = re.compile(
        r"sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,}"
    )
    for path in _committed_manifest_paths():
        text = path.read_text()
        assert not key_shaped.search(text), f"{path.name} contains a key-shaped literal"
        obj = json.loads(text)
        forbidden = {"api_key", "token", "secret", "apikey"}
        assert not (set(obj) & forbidden), f"{path.name} has a forbidden top-level secret key"


# ─── load_manifests: fail-loud on bad input ──────────────────────────────────


def test_load_manifests_rejects_extra_field(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(json.dumps({**_manifest(), "nope": 1}))
    with pytest.raises(ra.ReconcileError):
        ra.load_manifests(tmp_path)


def test_load_manifests_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ra.ReconcileError):
        ra.load_manifests(tmp_path)


def test_load_manifests_missing_dir_is_loud() -> None:
    with pytest.raises(ra.ReconcileError):
        ra.load_manifests("/nonexistent/agents/dir")


def test_load_manifests_reads_all_json(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_manifest(name="a")))
    (tmp_path / "b.json").write_text(json.dumps(_manifest(name="b")))
    (tmp_path / "ignore.txt").write_text("not a manifest")
    names = {m["name"] for m in ra.load_manifests(tmp_path)}
    assert names == {"a", "b"}


# ─── desired_diff: the truth table ───────────────────────────────────────────


def test_desired_diff_none_live_is_create() -> None:
    assert ra.desired_diff(_manifest(), None) == "create"


def test_desired_diff_identical_is_noop() -> None:
    m = _manifest()
    assert ra.desired_diff(m, _live_from(m)) == "noop"


def test_desired_diff_model_change_is_update() -> None:
    m = _manifest()
    live = _live_from(m)
    assert ra.desired_diff(_manifest(model="anthropic/claude-opus-4-9"), live) == "update"


def test_desired_diff_tool_added_is_update() -> None:
    m = _manifest()
    live = _live_from(m)
    changed = _manifest(tools=[{"type": "bash"}, {"type": "read"}, {"type": "edit"}])
    assert ra.desired_diff(changed, live) == "update"


def test_desired_diff_system_text_change_is_update() -> None:
    m = _manifest()
    live = _live_from(m)
    assert ra.desired_diff(_manifest(system="something else"), live) == "update"


def test_desired_diff_nested_reorder_is_noop() -> None:
    """Normalisation invariant: a manifest whose nested tool dict lists its keys in a
    different order than the live read view (and omits default-populated keys) must be
    `noop`, not phantom drift. Proves we compare canonical shapes, not raw text."""
    m = _manifest(tools=[{"type": "bash", "enabled": True}])
    # live read view: same tool but fully default-populated, keys in a different order
    live = _live_from(_manifest(tools=[{"type": "bash"}]))
    # sanity: the raw nested dicts differ in key set/order
    assert m["tools"][0] != live["tools"][0]
    assert ra.desired_diff(m, live) == "noop"


# ─── find_live_agent: duplicate-name guard ───────────────────────────────────


def _mock_request(status: int, payload: Any) -> Any:
    return mock.patch.object(ra, "_request", return_value=(status, payload))


def test_find_live_agent_zero_matches_is_none() -> None:
    with _mock_request(200, {"data": [], "has_more": False, "next_cursor": None}):
        assert ra.find_live_agent("https://x", "dev-implement", "k") is None


def test_find_live_agent_one_match_returns_it() -> None:
    row = {"id": "agent_1", "name": "dev-implement", "version": 4}
    with _mock_request(200, {"data": [row], "has_more": False, "next_cursor": None}):
        assert ra.find_live_agent("https://x", "dev-implement", "k") == row


def test_find_live_agent_reads_real_listresponse_data_envelope() -> None:
    """Regression: GET /v1/agents returns the ``ListResponse[T]`` envelope from
    src/aios/models/common.py — rows under 'data' (with 'has_more'/'next_cursor'),
    NOT 'items'. Reading 'items' aborted every reconcile/--check run with a false
    'response missing list' error. This pins the real envelope key."""
    row = {"id": "agent_real", "name": "dev-implement", "version": 9}
    envelope = {"data": [row], "has_more": False, "next_cursor": None}
    with _mock_request(200, envelope):
        assert ra.find_live_agent("https://x", "dev-implement", "k") == row


def test_find_live_agent_legacy_items_envelope_is_loud() -> None:
    """A response WITHOUT the 'data' key (e.g. the old/wrong 'items' shape) must
    fail loud rather than be silently treated as zero matches."""
    with (
        _mock_request(200, {"items": [{"id": "a", "name": "dev-implement"}]}),
        pytest.raises(ra.ReconcileError),
    ):
        ra.find_live_agent("https://x", "dev-implement", "k")


def test_find_live_agent_two_matches_raises() -> None:
    """>=2 exact-name matches → fail loud, NEVER silently pick one (prod has
    near-duplicate names, so this guard is load-bearing)."""
    rows = [
        {"id": "agent_1", "name": "autodev-resilience-lieutenant", "version": 1},
        {"id": "agent_2", "name": "autodev-resilience-lieutenant", "version": 1},
    ]
    with (
        _mock_request(200, {"data": rows, "has_more": False, "next_cursor": None}),
        pytest.raises(ra.ReconcileError),
    ):
        ra.find_live_agent("https://x", "autodev-resilience-lieutenant", "k")


def test_find_live_agent_non_2xx_is_loud() -> None:
    with _mock_request(500, {"detail": "boom"}), pytest.raises(ra.ReconcileError):
        ra.find_live_agent("https://x", "dev-implement", "k")


def test_find_live_agent_ignores_inexact_substring_match() -> None:
    """Re-asserts exact equality on the rows the endpoint returns, so a future
    prefix/substring regression can't mis-reconcile."""
    rows = [{"id": "agent_x", "name": "dev-implement-v2", "version": 1}]
    with _mock_request(200, {"data": rows, "has_more": False, "next_cursor": None}):
        assert ra.find_live_agent("https://x", "dev-implement", "k") is None


# ─── reconcile_agent: create-warning, check no-write, 409 ────────────────────


def test_reconcile_agent_create_emits_rewire_warning(capsys: pytest.CaptureFixture[str]) -> None:
    m = _manifest()
    with (
        mock.patch.object(ra, "find_live_agent", return_value=None),
        mock.patch.object(
            ra, "_post_agent", return_value={"id": "agent_NEW", "version": 1}
        ) as post,
    ):
        verdict = ra.reconcile_agent(m, base_url="https://x", api_key="k", check=False)
    assert verdict == "create"
    post.assert_called_once()
    out = capsys.readouterr().out
    assert "WARNING: created agent dev-implement id=agent_NEW" in out
    assert "re-registered to rewire" in out


def test_reconcile_agent_check_performs_no_write_on_create() -> None:
    m = _manifest()
    with (
        mock.patch.object(ra, "find_live_agent", return_value=None),
        mock.patch.object(ra, "_post_agent") as post,
        mock.patch.object(ra, "_put_agent") as put,
    ):
        verdict = ra.reconcile_agent(m, base_url="https://x", api_key="k", check=True)
    assert verdict == "create"
    post.assert_not_called()
    put.assert_not_called()


def test_reconcile_agent_check_performs_no_write_on_update() -> None:
    m = _manifest()
    live = _live_from(_manifest(model="anthropic/claude-opus-4-9"))
    with (
        mock.patch.object(ra, "find_live_agent", return_value=live),
        mock.patch.object(ra, "_post_agent") as post,
        mock.patch.object(ra, "_put_agent") as put,
    ):
        verdict = ra.reconcile_agent(m, base_url="https://x", api_key="k", check=True)
    assert verdict == "update"
    post.assert_not_called()
    put.assert_not_called()


def test_reconcile_agent_update_puts_with_live_version() -> None:
    m = _manifest(model="anthropic/claude-opus-4-9")
    live = _live_from(_manifest(), version=7, id="agent_LIVE")
    with mock.patch.object(ra, "_request") as req:
        # GET (find_live) then PUT
        req.side_effect = [
            (200, {"data": [live], "has_more": False, "next_cursor": None}),
            (200, {**live, "version": 8}),
        ]
        verdict = ra.reconcile_agent(m, base_url="https://x", api_key="k", check=False)
    assert verdict == "update"
    put_call = req.call_args_list[-1]
    assert put_call.args[0] == "PUT"
    assert "agent_LIVE" in put_call.args[1]
    assert put_call.kwargs["body"]["version"] == 7
    assert put_call.kwargs["body"]["model"] == "anthropic/claude-opus-4-9"


def test_reconcile_agent_noop_writes_nothing() -> None:
    m = _manifest()
    live = _live_from(m)
    with (
        mock.patch.object(ra, "find_live_agent", return_value=live),
        mock.patch.object(ra, "_post_agent") as post,
        mock.patch.object(ra, "_put_agent") as put,
    ):
        verdict = ra.reconcile_agent(m, base_url="https://x", api_key="k", check=False)
    assert verdict == "noop"
    post.assert_not_called()
    put.assert_not_called()


def test_reconcile_agent_put_409_is_loud() -> None:
    m = _manifest(model="anthropic/claude-opus-4-9")
    live = _live_from(_manifest(), version=3, id="agent_LIVE")
    with mock.patch.object(ra, "_request") as req:
        req.side_effect = [
            (200, {"data": [live], "has_more": False, "next_cursor": None}),
            (409, {"detail": "version mismatch"}),
        ]
        with pytest.raises(ra.ReconcileError):
            ra.reconcile_agent(m, base_url="https://x", api_key="k", check=False)


def test_reconcile_agent_put_version_must_increment() -> None:
    m = _manifest(model="anthropic/claude-opus-4-9")
    live = _live_from(_manifest(), version=5, id="agent_LIVE")
    with mock.patch.object(ra, "_request") as req:
        req.side_effect = [
            (200, {"data": [live], "has_more": False, "next_cursor": None}),
            (200, {**live, "version": 5}),  # did NOT increment
        ]
        with pytest.raises(ra.ReconcileError):
            ra.reconcile_agent(m, base_url="https://x", api_key="k", check=False)


# ─── main(): exit codes + --check drift verdict ──────────────────────────────


def _seed_dir(tmp_path: Path, *manifests: dict[str, Any]) -> Path:
    for m in manifests:
        (tmp_path / f"{m['name']}.json").write_text(json.dumps(m))
    return tmp_path


def test_main_missing_api_key_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_API_KEY", raising=False)
    monkeypatch.setenv("AIOS_URL", "https://x")
    assert ra.main([]) == 2


def test_main_missing_url_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.delenv("AIOS_URL", raising=False)
    assert ra.main([]) == 2


def test_main_check_returns_zero_when_in_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _manifest(name="a")
    _seed_dir(tmp_path, m)
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://x")
    with mock.patch.object(ra, "find_live_agent", return_value=_live_from(m)):
        rc = ra.main(["--check", "--manifest-dir", str(tmp_path)])
    assert rc == 0


def test_main_check_returns_nonzero_on_drift_and_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _manifest(name="a")
    _seed_dir(tmp_path, m)
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://x")
    live = _live_from(_manifest(name="a", model="anthropic/claude-opus-4-9"))
    with (
        mock.patch.object(ra, "find_live_agent", return_value=live),
        mock.patch.object(ra, "_put_agent") as put,
        mock.patch.object(ra, "_post_agent") as post,
    ):
        rc = ra.main(["--check", "--manifest-dir", str(tmp_path)])
    assert rc == 1
    put.assert_not_called()
    post.assert_not_called()


def test_main_only_filters_to_named_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_dir(tmp_path, _manifest(name="a"), _manifest(name="b"))
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://x")
    seen: list[str] = []

    def _stub(manifest: dict[str, Any], **_: Any) -> str:
        seen.append(manifest["name"])
        return "noop"

    with mock.patch.object(ra, "reconcile_agent", _stub):
        rc = ra.main(["--only", "b", "--manifest-dir", str(tmp_path)])
    assert rc == 0
    assert seen == ["b"]


def test_main_unknown_only_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_dir(tmp_path, _manifest(name="a"))
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://x")
    assert ra.main(["--only", "nope", "--manifest-dir", str(tmp_path)]) == 2


def test_main_aborts_when_a_manifest_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_dir(tmp_path, _manifest(name="a"))
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://x")

    def _boom(manifest: dict[str, Any], **_: Any) -> str:
        raise ra.ReconcileError("kaboom")

    with mock.patch.object(ra, "reconcile_agent", _boom):
        assert ra.main(["--manifest-dir", str(tmp_path)]) == 1


def test_main_real_committed_dir_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """--manifest-dir defaults to infra/agents and the committed manifests load."""
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://x")
    seen: list[str] = []

    def _stub(manifest: dict[str, Any], **_: Any) -> str:
        seen.append(manifest["name"])
        return "noop"

    # run from repo root so the default relative dir resolves
    monkeypatch.chdir(_REPO_ROOT)
    with mock.patch.object(ra, "reconcile_agent", _stub):
        rc = ra.main([])
    assert rc == 0
    assert "dev-implement" in seen


# ─── Action drift guards ─────────────────────────────────────────────────────


def test_workflow_triggers_on_manifest_path() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert "branches: [master]" in text
    assert re.search(r"paths:\s*\n\s*-\s*\"infra/agents/\*\*\"", text)


def test_workflow_least_privilege_and_secret() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert re.search(r"permissions:\s*\n\s*contents:\s*read", text)
    assert "${{ secrets.AIOS_API_KEY }}" in text
    assert "AIOS_URL: https://api.aios.eumemic.ai" in text


def test_workflow_concurrency_serialises() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert "group: reconcile-agents" in text
    assert "cancel-in-progress: false" in text


def test_workflow_runs_unit_gate_before_reconcile() -> None:
    text = _WORKFLOW_PATH.read_text()
    gate_at = text.find("tests/unit/test_reconcile_agents.py")
    reconcile_at = text.find("python scripts/reconcile_agents.py")
    assert 0 < gate_at < reconcile_at


def test_no_hardcoded_api_key_in_action_or_script() -> None:
    for path in (_WORKFLOW_PATH, _SCRIPT_PATH):
        text = path.read_text()
        assert not re.search(r"AIOS_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", text)


def test_api_key_is_env_only_no_cli_flag() -> None:
    text = _SCRIPT_PATH.read_text()
    assert "--api-key" not in text
    assert 'os.environ.get("AIOS_API_KEY")' in text
