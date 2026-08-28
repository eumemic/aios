"""Behavioral e2e for the aios-browser image: real Chromium, driven through
the same ``docker exec browser-cli`` transport the worker uses.

Two containers, both provisioned with the production flag set (seccomp
profile, resource caps, --init, --ipc private, no-new-privileges, plane
mount — see ``tests/e2e/browser_image_common.py``):

* ``knob`` — carries ``AIOS_BROWSER_DRIVER_ALLOW_PRIVATE_NAV=1`` (the
  hermetic-test knob; unsettable through aios because the browser spec pins
  ``environment={}``) plus a loopback HTTP fixture served from the plane, and
  hosts the action round-trips, the takeover lifecycle, and the RSS trip-wire.
* ``strict`` — no knob: proves the navigate guard's deny matrix as shipped.

Like the contract suite this file is aios-import-free and opt-in: it skips
when ``AIOS_SANDBOX_BROWSER_IMAGE`` is unset.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import needs_docker
from tests.e2e.browser_image_common import (
    IMAGE,
    invoke,
    make_plane,
    run,
    start_browser_container,
    wait_ready,
)

pytestmark = [
    needs_docker,
    pytest.mark.docker,
    pytest.mark.skipif(
        not IMAGE, reason="AIOS_SANDBOX_BROWSER_IMAGE not set; browser behavior suite is opt-in"
    ),
]

_FIXTURE_PORT = 8907
_FIXTURE_URL = f"http://127.0.0.1:{_FIXTURE_PORT}/"

# The loopback fixture site. ``index.html`` exercises every action op the
# suite drives; ``popup.html`` is the popup/staleness target; ``file.bin`` the
# download payload. Title mutations are the observable back-channel: the
# response's ``title`` is read post-settle, so DOM-event side effects
# (form submit, scroll) become assertable through the wire contract alone.
_INDEX_HTML = """<!doctype html><html><head><title>fixture</title></head><body>
<h1>Behavior fixture</h1>
<button id="add" onclick="this.textContent='clicked-ok'">Add item</button>
<form onsubmit="event.preventDefault();document.title='typed-'+document.getElementById('t').value">
  <input id="t" aria-label="Query">
</form>
<input type="password" aria-label="Secret">
<select aria-label="Color"><option value="r">red</option><option value="g">green</option></select>
<a href="/file.bin" download="file.bin">Download file</a>
<button id="pop" onclick="window.open('/popup.html','_blank')">Open popup</button>
<div style="height:4000px"></div>
<script>addEventListener('scroll',()=>{document.title='scrolled-'+Math.round(scrollY)})</script>
</body></html>"""
_POPUP_HTML = (
    "<!doctype html><html><head><title>popup</title></head><body><h1>Popup page</h1></body></html>"
)


def _action(container: str, op: str, **args: Any) -> dict[str, Any]:
    """One action op for the suite's single agent session ``s1``."""
    return invoke(container, {"op": op, "session_id": "s1", "args": args, "timeout_ms": 30_000})


def _ref_of(snapshot: str, name: str) -> str:
    """The [ref=eN] handle of the snapshot line naming *name*."""
    for line in snapshot.splitlines():
        if name in line:
            m = re.search(r"\[ref=(e\d+)\]", line)
            if m:
                return m.group(1)
    pytest.fail(f"no ref for {name!r} in snapshot:\n{snapshot}")


def _write_fixture_site(plane: Path) -> None:
    """The fixture site lives on the plane bind mount (a throwaway test
    plane), so it needs no in-container writes — only the http.server exec."""
    www = plane / "www"
    www.mkdir()
    www.chmod(0o777)
    (www / "index.html").write_text(_INDEX_HTML)
    (www / "popup.html").write_text(_POPUP_HTML)
    (www / "file.bin").write_bytes(b"download-payload-" * 64)


def _serve_fixture_site(container: str) -> None:
    r = run(
        [
            *("docker", "exec", "--detach", container),
            *("python3", "-m", "http.server", str(_FIXTURE_PORT)),
            *("--bind", "127.0.0.1", "--directory", "/workspace/www"),
        ],
        timeout=30,
    )
    assert r.returncode == 0, f"fixture server start failed: {r.stderr}"
    deadline = time.monotonic() + 15
    probe = f"import urllib.request; urllib.request.urlopen('{_FIXTURE_URL}', timeout=2)"
    while time.monotonic() < deadline:
        if run(["docker", "exec", container, "python3", "-c", probe], timeout=15).returncode == 0:
            return
        time.sleep(0.5)
    pytest.fail("in-container fixture server never came up")


@pytest.fixture(scope="module")
def knob(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path]]:
    """(container_name, plane_dir) with the private-nav knob + fixture site."""
    plane = make_plane(tmp_path_factory.mktemp("browser-knob"))
    _write_fixture_site(plane)
    name = f"aios-browser-behav-{uuid.uuid4().hex[:8]}"
    start_browser_container(IMAGE, plane, name, env={"AIOS_BROWSER_DRIVER_ALLOW_PRIVATE_NAV": "1"})
    try:
        wait_ready(name)
        _serve_fixture_site(name)
        yield name, plane
    finally:
        run(["docker", "rm", "--force", name], timeout=30)


@pytest.fixture(scope="module")
def strict(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A knobless container: the navigate guard as shipped."""
    plane = make_plane(tmp_path_factory.mktemp("browser-strict"))
    name = f"aios-browser-strict-{uuid.uuid4().hex[:8]}"
    start_browser_container(IMAGE, plane, name)
    try:
        wait_ready(name)
        yield name
    finally:
        run(["docker", "rm", "--force", name], timeout=30)


# -- action round-trips (knob container) ---------------------------------------


def test_navigate_click_type_scroll_roundtrip(knob: tuple[str, Path]) -> None:
    container, _ = knob

    nav = _action(container, "navigate", url=_FIXTURE_URL)
    assert nav["ok"], nav["error"]
    assert nav["url"] == _FIXTURE_URL
    assert "[ref=" in nav["snapshot"]

    # click mutates the button's own accessible name — visible in the next snapshot
    ref = _ref_of(nav["snapshot"], "Add item")
    clicked = _action(container, "click", ref=ref, description="add an item")
    assert clicked["ok"], clicked["error"]
    assert "clicked-ok" in clicked["snapshot"]

    # type + submit drives the form; the title mutation proves the keystrokes landed
    query_ref = _ref_of(clicked["snapshot"], "Query")
    typed = _action(container, "type", ref=query_ref, text="hello", submit=True, description="q")
    assert typed["ok"], typed["error"]
    assert typed["title"] == "typed-hello"

    # scroll down: positive scrollY proves the wheel sign (down = positive delta)
    scrolled = _action(container, "scroll", direction="down")
    assert scrolled["ok"], scrolled["error"]
    m = re.fullmatch(r"scrolled-(\d+)", scrolled["title"] or "")
    assert m and int(m.group(1)) > 0, f"wheel sign wrong or scroll inert: {scrolled['title']!r}"

    # the remaining pointer ops accept the documented shapes. Refs are
    # CURRENT-GENERATION: every action re-snapshots and supersedes earlier
    # handles, so each ref must come from the immediately preceding response.
    hovered = _action(container, "hover", ref=_ref_of(scrolled["snapshot"], "clicked-ok"))
    assert hovered["ok"], hovered["error"]
    dragged = invoke(
        container,
        {
            "op": "drag",
            "session_id": "s1",
            "args": {"from": {"x": 100, "y": 100}, "to": {"x": 200, "y": 200}},
            "timeout_ms": 30_000,
        },
    )
    assert dragged["ok"], dragged["error"]
    clicked_xy = _action(container, "click_xy", x=10, y=10, description="corner")
    assert clicked_xy["ok"], clicked_xy["error"]

    sel_ref = _ref_of(clicked_xy["snapshot"], "Color")
    selected = _action(container, "select_option", ref=sel_ref, values=["g"])
    assert selected["ok"], selected["error"]


def test_screenshot_lands_in_plane_shots(knob: tuple[str, Path]) -> None:
    container, plane = knob
    _action(container, "navigate", url=_FIXTURE_URL)
    shot = _action(container, "screenshot")
    assert shot["ok"], shot["error"]
    path = shot["shot_path"]
    assert path and path.startswith("shots/")
    host_file = plane / path
    assert host_file.exists() and host_file.stat().st_size > 0
    assert host_file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_download_persists_to_plane(knob: tuple[str, Path]) -> None:
    container, plane = knob
    nav = _action(container, "navigate", url=_FIXTURE_URL)
    ref = _ref_of(nav["snapshot"], "Download file")
    clicked = _action(container, "click", ref=ref, description="download")
    assert clicked["ok"], clicked["error"]
    # Persisted under a ULID prefix (collision-proofing) + the suggested name.
    deadline = time.monotonic() + 15
    persisted: list[Path] = []
    while time.monotonic() < deadline and not persisted:
        persisted = sorted((plane / "downloads").glob("*-file.bin"))
        if not persisted:
            time.sleep(0.5)
    assert persisted, "download never persisted to the plane downloads dir"
    assert persisted[0].read_bytes().startswith(b"download-payload-")


def test_ref_staleness_and_unknown_ref(knob: tuple[str, Path]) -> None:
    container, _ = knob
    nav = _action(container, "navigate", url=_FIXTURE_URL)
    old_ref = _ref_of(nav["snapshot"], "Add item")
    _action(container, "navigate", url=_FIXTURE_URL + "popup.html")
    stale = _action(container, "click", ref=old_ref, description="stale")
    assert not stale["ok"] and stale["error"]["code"] == "stale_snapshot"
    assert stale["snapshot"], "ok:false must still carry a fresh snapshot (self-correction)"
    unknown = _action(container, "click", ref="e99999", description="never issued")
    assert not unknown["ok"] and unknown["error"]["code"] == "no_such_ref"


def test_password_guardrail(knob: tuple[str, Path]) -> None:
    container, _ = knob
    nav = _action(container, "navigate", url=_FIXTURE_URL)
    pw_ref = _ref_of(nav["snapshot"], "Secret")
    refused = _action(container, "type", ref=pw_ref, text="hunter2", description="pw")
    assert not refused["ok"] and refused["error"]["code"] == "not_interactable"


def test_popup_is_auto_followed(knob: tuple[str, Path]) -> None:
    container, _ = knob
    nav = _action(container, "navigate", url=_FIXTURE_URL)
    ref = _ref_of(nav["snapshot"], "Open popup")
    popped = _action(container, "click", ref=ref, description="open popup")
    assert popped["ok"], popped["error"]
    assert popped["url"].endswith("/popup.html")
    assert len(popped["tabs"]) == 2
    active = [t for t in popped["tabs"] if t["active"]]
    assert len(active) == 1 and active[0]["url"].endswith("/popup.html")


# -- navigate guard as shipped (strict container) ------------------------------


@pytest.mark.parametrize(
    ("url", "expected_message"),
    [
        # Address rows must carry the guard's OWN message: any connect failure
        # also maps to navigation_failed, so without the message assert these
        # rows would stay green with the address check deleted.
        (_FIXTURE_URL, "non-public address"),  # loopback, no knob
        ("http://169.254.169.254/latest/meta-data/", "non-public address"),  # cloud metadata
        ("https://[::1]/", "non-public address"),  # v6 loopback
        ("file:///etc/passwd", "only http(s) URLs"),  # scheme
        ("data:text/html,hi", "only http(s) URLs"),  # scheme
    ],
)
def test_navigate_guard_denies(strict: str, url: str, expected_message: str) -> None:
    denied = _action(strict, "navigate", url=url)
    assert not denied["ok"], f"guard let {url!r} through"
    assert denied["error"]["code"] == "navigation_failed"
    assert expected_message in denied["error"]["message"], denied["error"]


# -- takeover lifecycle over the wire ------------------------------------------


def test_takeover_lifecycle_screencast_and_input(knob: tuple[str, Path]) -> None:
    container, plane = knob

    # Land on the fixture and focus the query input (so spooled text has a target).
    nav = _action(container, "navigate", url=_FIXTURE_URL)
    query_ref = _ref_of(nav["snapshot"], "Query")
    focused = _action(container, "click", ref=query_ref, description="focus input")
    assert focused["ok"], focused["error"]

    grant = f"g-{uuid.uuid4().hex[:6]}"
    opened = invoke(
        container,
        {
            "op": "takeover_open",
            "session_id": "s1",
            "args": {"grant_id": grant, "reason": "e2e"},
            "timeout_ms": 45_000,
        },
        wrapper_s=50,
    )
    assert opened["ok"], opened["error"]
    epoch = opened["epoch"]
    assert epoch > 0
    assert opened["data"]["target"]["url"] == _FIXTURE_URL
    # page-blind top level while the gate is closed
    assert opened["snapshot"] is None and opened["tabs"] == []
    assert opened["url"] is None and opened["title"] is None

    # agent actions are refused, page-blind — including url/title (a login URL
    # can itself carry secrets in its query string)
    blocked = _action(container, "snapshot")
    assert not blocked["ok"] and blocked["error"]["code"] == "takeover_active"
    assert blocked["snapshot"] is None and blocked["tabs"] == []
    assert blocked["url"] is None and blocked["title"] is None

    # status is the ONE op the gate does not block (the product polls it during
    # takeovers) — and it stays page-blind beyond url/title.
    status = invoke(container, {"op": "status", "timeout_ms": 10_000}, wrapper_s=15)
    assert status["ok"], status["error"]
    assert status["snapshot"] is None and status["tabs"] == []

    # idempotent re-open: pure echo of the original epoch
    echo = invoke(
        container,
        {
            "op": "takeover_open",
            "session_id": "s1",
            "args": {"grant_id": grant},
            "timeout_ms": 45_000,
        },
        wrapper_s=50,
    )
    assert echo["ok"] and echo["epoch"] == epoch

    # the screencast publishes an atomic manifest naming a real frame
    manifest_path = plane / "frames" / "manifest.json"
    deadline = time.monotonic() + 20
    manifest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            break
        time.sleep(0.5)
    assert manifest, "screencast never published a manifest"
    assert manifest["epoch"] == epoch
    assert manifest["origin"] == _FIXTURE_URL.rstrip("/")
    assert manifest["security"] == "insecure"  # http fixture — derived from scheme
    assert manifest["file"] == f"frame-{manifest['seq']}.jpg" and "/" not in manifest["file"]
    assert (plane / "frames" / manifest["file"]).exists()

    # spooled human input drives the page: text lands in the focused input,
    # Enter submits, and the title mutation is observed after close. A second
    # batch under a STALE epoch must be dropped. Written BEFORE the slower RSS
    # probe so the takeover never counts as unclaimed (the driver self-closes
    # an input-less, heartbeat-less takeover after 60s).
    spool = plane / "input" / "spool.jsonl"
    batches = [
        {
            "grant_id": grant,
            "epoch": epoch,
            "seq": 1,
            "events": [
                {"type": "text", "text": "human"},
                {"type": "key_down", "key": "Enter"},
                {"type": "key_up", "key": "Enter"},
            ],
        },
        {
            "grant_id": grant,
            "epoch": epoch - 1,
            "seq": 2,
            "events": [{"type": "text", "text": "STALE"}],
        },
    ]
    with spool.open("a") as f:
        f.write("".join(json.dumps(b) + "\n" for b in batches))
    time.sleep(2.0)  # tailer polls every 50ms; give injection + submit time to land

    # RSS trip-wire under an active screencast (plan R1): < 1.5 GiB, inside
    # the production 2 GiB cap the container runs under.
    stats = run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container], timeout=30
    )
    assert stats.returncode == 0, stats.stderr
    used = stats.stdout.split("/")[0].strip()
    m = re.fullmatch(r"([\d.]+)\s*(KiB|MiB|GiB)", used)
    assert m, f"unparseable mem usage {used!r}"
    scale = {"KiB": 1 / (1024 * 1024), "MiB": 1 / 1024, "GiB": 1.0}[m.group(2)]
    gib = float(m.group(1)) * scale
    assert gib < 1.5, f"browser container RSS {gib:.2f} GiB exceeds the 1.5 GiB trip-wire"

    closed = invoke(
        container,
        {"op": "takeover_close", "args": {"grant_id": grant, "outcome": "done"}},
        wrapper_s=40,
    )
    assert closed["ok"], closed["error"]
    assert closed["epoch"] > epoch  # rotated again on close
    assert closed["url"] is not None
    assert closed["snapshot"], "handback must carry a fresh snapshot"
    assert closed["shot_path"] and closed["shot_path"].startswith("shots/")
    assert (plane / closed["shot_path"]).exists()
    assert "signed_in_hosts" in closed["data"]

    # replay: a redriven close returns the cached handback
    replay = invoke(
        container,
        {"op": "takeover_close", "args": {"grant_id": grant, "outcome": "done"}},
        wrapper_s=40,
    )
    assert replay["ok"] and replay["url"] == closed["url"]

    # unknown grant → no_takeover; the agent can act again
    unknown = invoke(
        container,
        {"op": "takeover_close", "args": {"grant_id": "g-nope"}},
        wrapper_s=40,
    )
    assert not unknown["ok"] and unknown["error"]["code"] == "no_takeover"

    # The agent can act again — and the re-observed page proves the spooled
    # input drove it: the text landed in the focused input and Enter submitted
    # the form (title mutation), while the stale-epoch batch was dropped.
    acting = _action(container, "snapshot")
    assert acting["ok"], acting["error"]
    assert acting["title"] == "typed-human", (
        f"spooled input never drove the page (title={acting['title']!r}) — "
        "or the stale-epoch batch was applied"
    )
