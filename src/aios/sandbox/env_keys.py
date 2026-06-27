"""Single source of truth for the env-var **key names** the harness injects
into every sandbox.

This module is intentionally dependency-free — it imports nothing from
``aios`` (and nothing heavy like ``cryptography``) — so both the sandbox env
*producers* and ``aios.models.vaults`` can import it without an import cycle:

* importing ``sandbox.setup`` from the model layer would cycle via
  ``aios.config``;
* importing ``sandbox.egress_ca`` would drag cryptography's x509 machinery
  into every ``models`` import.

The producers (``sandbox.setup`` ``WORKSPACE_RUNTIME_ENV``,
``sandbox.egress_ca`` ``TRUST_STORE_ENV``, ``sandbox.spec`` ``merged_env``
trailers, ``workflows.run_sandbox`` exec preamble) build their env dicts /
shell preambles keyed off these names, and
``models.vaults.RESERVED_SANDBOX_ENV_KEYS`` is derived from the same constants.
So an ``environment_variable`` credential can never claim a harness-injected
key as its ``secret_name``, and adding a new injected key in one place flows
into the blocklist with no second edit — the invariant is held by construction,
not by a drift test.
"""

from __future__ import annotations

# --- sandbox.setup WORKSPACE_RUNTIME_ENV ------------------------------------
# The absolute system PATH baked into every sandbox.
PATH_ENV_KEY = "PATH"
WORKSPACE_RUNTIME_ENV_KEYS: frozenset[str] = frozenset({PATH_ENV_KEY})

# --- sandbox.egress_ca TRUST_STORE_ENV --------------------------------------
# CA-bundle pointers so OpenSSL clients and Node trust the aios egress CA.
SSL_CERT_FILE_ENV_KEY = "SSL_CERT_FILE"
REQUESTS_CA_BUNDLE_ENV_KEY = "REQUESTS_CA_BUNDLE"
NODE_EXTRA_CA_CERTS_ENV_KEY = "NODE_EXTRA_CA_CERTS"
TRUST_STORE_ENV_KEYS: frozenset[str] = frozenset(
    {SSL_CERT_FILE_ENV_KEY, REQUESTS_CA_BUNDLE_ENV_KEY, NODE_EXTRA_CA_CERTS_ENV_KEY}
)

# --- sandbox.spec merged_env trailers ---------------------------------------
# Tool-broker wiring + the session id exposed for sandbox_command fires.
TOOL_BROKER_URL_ENV_KEY = "TOOL_BROKER_URL"
TOOL_BROKER_SECRET_ENV_KEY = "TOOL_BROKER_SECRET"
AIOS_SESSION_ID_ENV_KEY = "AIOS_SESSION_ID"
SPEC_TRAILER_ENV_KEYS: frozenset[str] = frozenset(
    {TOOL_BROKER_URL_ENV_KEY, TOOL_BROKER_SECRET_ENV_KEY, AIOS_SESSION_ID_ENV_KEY}
)

# --- workflows.run_sandbox exec preamble ------------------------------------
# Exported into the run bash command's environment (not injected at container
# create time) so an author can pass ``$AIOS_IDEMPOTENCY_KEY`` to an external
# service for crash-re-drive dedupe.
AIOS_RUN_ID_ENV_KEY = "AIOS_RUN_ID"
AIOS_IDEMPOTENCY_KEY_ENV_KEY = "AIOS_IDEMPOTENCY_KEY"
RUN_PREAMBLE_ENV_KEYS: frozenset[str] = frozenset(
    {AIOS_RUN_ID_ENV_KEY, AIOS_IDEMPOTENCY_KEY_ENV_KEY}
)

# The complete set of harness-injected env-var key names: the union of every
# producer's declared key set. ``models.vaults`` uses exactly this as its
# reserved blocklist.
RESERVED_SANDBOX_ENV_KEYS: frozenset[str] = (
    WORKSPACE_RUNTIME_ENV_KEYS
    | TRUST_STORE_ENV_KEYS
    | SPEC_TRAILER_ENV_KEYS
    | RUN_PREAMBLE_ENV_KEYS
)
