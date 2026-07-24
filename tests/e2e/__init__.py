"""End-to-end test package configuration."""

import os

# E2E fixtures intentionally use loopback MCP targets backed by local test services.
# One operator allowlist governs BOTH the write-boundary target validator and the
# connection-time PinnedTransport (drift fix, PR #1931 review).
os.environ.setdefault("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", "127.0.0.1,localhost")
