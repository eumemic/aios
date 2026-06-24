"""Shared test setup.

``HttpConnector`` reads ``AIOS_URL`` / ``AIOS_RUNTIME_TOKEN`` at
``__init__`` time, so any test that constructs an ``SmsConnector`` needs
them in env. Set at import time so module-level instances also see them.
"""

from __future__ import annotations

import os

os.environ.setdefault("AIOS_URL", "http://test")
os.environ.setdefault("AIOS_RUNTIME_TOKEN", "aios_runtime_test")
