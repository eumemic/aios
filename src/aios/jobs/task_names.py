"""Names of jobs registered by the worker harness.

Keep this module dependency-free.  Queue infrastructure and worker boot checks
need a shared description of the required registrations without importing the
worker's execution graph merely to inspect the names.
"""

from __future__ import annotations

REQUIRED_HARNESS_TASKS: frozenset[str] = frozenset(
    {
        "harness.wake_session",
        "harness.wake_workflow",
        "harness.run_trigger",
    }
)
