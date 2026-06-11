"""E2E: every provisioned sandbox trusts the worker's egress CA.

Provisions through the real registry path (so the ``install_egress_ca``
wiring in ``_provision`` is what's under test, not a hand-called setup
step) and asserts issue #875's acceptance: ``curl https://<host>``
inside the sandbox verifies a leaf signed by the aios CA.

The leaf is deliberately minted from a *fresh* :class:`EgressCA` built
from the same seed — not the process-cached instance that installed the
cert — so the cross-process contract (worker A installs its cert copy,
worker B mints the leaf; subject + keypair deterministic, serial not)
is exercised end to end.
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization

from aios.harness import runtime
from aios.sandbox.egress_ca import (
    CA_CERT_SANDBOX_PATH,
    EGRESS_CA_HKDF_INFO,
    SYSTEM_CA_BUNDLE_PATH,
    EgressCA,
)
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, bash, first_tool_result
from tests.helpers.tls import mint_leaf

pytestmark = pytest.mark.docker

TLS_HOST = "api.example.com"
TLS_PORT = 8443

# Detach the server's stdio (a backgrounded process inheriting the
# docker-exec pipes would keep the exec open until its kill backstop)
# and poll readiness instead of racing the bind.
_SERVE_AND_CURL = f"""
cat > /tmp/tls-srv.py <<'PY'
import http.server, ssl
srv = http.server.HTTPServer(("127.0.0.1", {TLS_PORT}), http.server.SimpleHTTPRequestHandler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain("/tmp/tls-leaf.pem", "/tmp/tls-key.pem")
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
srv.serve_forever()
PY
python3 /tmp/tls-srv.py >/tmp/tls-srv.log 2>&1 </dev/null &
# Worst case ~36s (30 x (1s curl + 0.2s sleep)) — well under the bash
# tool's 120s ceiling even with slow container startup in CI.
for i in $(seq 30); do
  if curl -sS --max-time 1 --resolve {TLS_HOST}:{TLS_PORT}:127.0.0.1 \\
       https://{TLS_HOST}:{TLS_PORT}/ >/dev/null 2>/tmp/tls-curl.err; then
    echo TLS_VERIFY_OK
    exit 0
  fi
  sleep 0.2
done
cat /tmp/tls-srv.log /tmp/tls-curl.err >&2
exit 1
"""


@needs_docker
class TestEgressCATrust:
    async def test_ca_installed_and_folded_into_system_bundle(
        self, docker_harness: Harness
    ) -> None:
        """The drop-in exists and update-ca-certificates folded it into
        the aggregate bundle SSL_CERT_FILE/REQUESTS_CA_BUNDLE point at —
        matched by the cert's first base64 payload line, not exact PEM
        (the bundle strips nothing here, but anchoring on content keeps
        the assertion serial-agnostic)."""
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[
                        bash(
                            # The non-empty guard on the anchor line keeps the
                            # grep from degrading to vacuous (grep -qF "" matches
                            # anything) if the drop-in were ever written mangled.
                            f'anchor="$(sed -n 2p {CA_CERT_SANDBOX_PATH})" && '
                            f'test -n "$anchor" && '
                            f'grep -qF "$anchor" {SYSTEM_CA_BUNDLE_PATH} && '
                            f"echo BUNDLE_CONTAINS_CA && "
                            f"echo SSL_CERT_FILE=$SSL_CERT_FILE && "
                            f"echo REQUESTS_CA_BUNDLE=$REQUESTS_CA_BUNDLE && "
                            f"echo NODE_EXTRA_CA_CERTS=$NODE_EXTRA_CA_CERTS"
                        )
                    ],
                ),
                assistant("Done."),
            ]
        )
        session = await docker_harness.start("test", tools=["bash"])
        await docker_harness.run_until_idle(session.id)

        content = str(first_tool_result(await docker_harness.events(session.id)).get("content", ""))
        assert "BUNDLE_CONTAINS_CA" in content
        assert f"SSL_CERT_FILE={SYSTEM_CA_BUNDLE_PATH}" in content
        assert f"REQUESTS_CA_BUNDLE={SYSTEM_CA_BUNDLE_PATH}" in content
        assert f"NODE_EXTRA_CA_CERTS={CA_CERT_SANDBOX_PATH}" in content

    async def test_curl_verifies_leaf_signed_by_aios_ca(self, docker_harness: Harness) -> None:
        """Issue #875 acceptance: an in-sandbox TLS server presents a
        leaf for an allowlisted-host-shaped name, signed by a fresh
        EgressCA instance from the worker's seed; curl verifies it
        against the system trust store."""
        seed = runtime.require_crypto_box().derive_subkey_bytes(EGRESS_CA_HKDF_INFO)
        leaf, key = mint_leaf(EgressCA(seed), TLS_HOST)
        leaf_pem = leaf.public_bytes(serialization.Encoding.PEM).decode("ascii")
        key_pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode("ascii")

        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[
                        bash(
                            f"printf '%s' '{leaf_pem}' > /tmp/tls-leaf.pem && "
                            f"printf '%s' '{key_pem}' > /tmp/tls-key.pem && "
                            f"{_SERVE_AND_CURL}"
                        )
                    ],
                ),
                assistant("Done."),
            ]
        )
        session = await docker_harness.start("test", tools=["bash"])
        await docker_harness.run_until_idle(session.id)

        content = str(first_tool_result(await docker_harness.events(session.id)).get("content", ""))
        assert "TLS_VERIFY_OK" in content, content
