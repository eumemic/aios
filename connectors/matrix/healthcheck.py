from __future__ import annotations

import json
import os
import urllib.request

host, _, port = os.environ.get("MATRIX_LISTEN_ADDR", "0.0.0.0:29328").rpartition(":")
if host in {"", "0.0.0.0", "::"}:
    host = "127.0.0.1"
request = urllib.request.Request(
    f"http://{host}:{port}/_matrix/app/v1/ping?access_token={os.environ['MATRIX_HS_TOKEN']}",
    data=json.dumps({"transaction_id": "container-healthcheck"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
