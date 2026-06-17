"""Shared test fixtures + env wiring for the slack connector tests.

``HttpConnector`` reads ``AIOS_URL`` / ``AIOS_RUNTIME_TOKEN`` at
``__init__`` time, so set them at import time (before the connector is
constructed) rather than in a fixture.
"""

from __future__ import annotations

import os

os.environ.setdefault("AIOS_URL", "http://test")
os.environ.setdefault("AIOS_RUNTIME_TOKEN", "aios_runtime_test")
