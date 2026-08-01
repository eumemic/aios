"""Lane activation — the ``lane_activate`` workflow and supporting models.

A *lane* is a declarative bundle (workflow + launcher agent + launcher session +
cron trigger) whose desired state is checked into source control as a
``<lane>.lock.json`` file.  The ``lane_activate`` workflow reads the lock at a
given ``merge_sha``, diffs it against the live aios objects, and converges the
live state — creating new objects or updating existing ones with optimistic
concurrency, never deleting.
"""

from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT
from aios.lanes.models import (
    ActivationOutcome,
    ActivationResult,
    LaneLock,
    LockCronTrigger,
    LockLauncherAgent,
    LockLauncherSession,
    LockProvenance,
    LockWorkflow,
    ObjectDelta,
)

__all__ = [
    "LANE_ACTIVATE_SCRIPT",
    "ActivationOutcome",
    "ActivationResult",
    "LaneLock",
    "LockCronTrigger",
    "LockLauncherAgent",
    "LockLauncherSession",
    "LockProvenance",
    "LockWorkflow",
    "ObjectDelta",
]
