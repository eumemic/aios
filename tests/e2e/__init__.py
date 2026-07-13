"""End-to-end test package configuration."""

import os

# E2E fixtures intentionally use loopback MCP targets backed by local test services.
os.environ.setdefault("AIOS_TARGET_URL_ALLOW_HOSTS", "127.0.0.1,localhost")
