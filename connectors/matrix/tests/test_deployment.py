from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_registration_namespace_is_exact_and_no_e2ee_flags() -> None:
    registration = yaml.safe_load((ROOT / "registration.yaml").read_text())
    assert registration == {
        "id": "aios-matrix",
        "url": "http://matrix:29328",
        "as_token": "CHANGE_ME_AS_TOKEN",
        "hs_token": "CHANGE_ME_HS_TOKEN",
        "sender_localpart": "_aios",
        "rate_limited": True,
        "namespaces": {
            "users": [{"exclusive": True, "regex": r"^@_aios_agent_[a-z0-9]+:your\.server$"}],
            "aliases": [],
            "rooms": [],
        },
    }


def test_synapse_overlay_closes_federation_and_enables_retention() -> None:
    config = yaml.safe_load((ROOT / "synapse.yaml").read_text())
    assert config["app_service_config_files"] == ["/data/aios-matrix-registration.yaml"]
    assert config["federation_domain_whitelist"] == {}
    assert config["allow_profile_lookup_over_federation"] is False
    assert config["forget_rooms_on_leave"] is True
    assert config["forgotten_room_retention_period"] == "28d"


class _HealthyAppservice(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def test_healthcheck_requires_connection_serving_heartbeat(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthyAppservice)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    heartbeat = tmp_path / "connector-heartbeat"
    env = {
        **os.environ,
        "AIOS_CONNECTOR_HEARTBEAT_PATH": str(heartbeat),
        "MATRIX_HS_TOKEN": "test-token",
        "MATRIX_LISTEN_ADDR": f"127.0.0.1:{server.server_port}",
    }
    try:
        # A responsive appservice does not prove that any active connection's
        # inbound worker is established.  Missing SDK heartbeat must fail.
        result = subprocess.run(
            [sys.executable, str(ROOT / "healthcheck.py")],
            env=env,
            check=False,
        )
        assert result.returncode != 0

        heartbeat.write_text("{malformed")
        result = subprocess.run(
            [sys.executable, str(ROOT / "healthcheck.py")],
            env=env,
            check=False,
        )
        assert result.returncode != 0

        # Match the structurally valid payload emitted by the SDK heartbeat
        # publisher once discovery has completed and no transport is unhealthy.
        heartbeat.write_text(
            json.dumps(
                {
                    "healthy_connection_ids": ["matrix-test-connection"],
                    "unhealthy_connection_ids": [],
                },
                sort_keys=True,
            )
        )
        result = subprocess.run(
            [sys.executable, str(ROOT / "healthcheck.py")],
            env=env,
            check=False,
        )
        assert result.returncode == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
