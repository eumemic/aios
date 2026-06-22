"""Drift-detection tooling that runs over the source tree itself.

These modules scan aios's own code/comments for signals that rot — references
that lie once the thing they point at changes. They are pure/offline (no network,
no DB); the online half (e.g. checking an issue's live state) lives in the
``.github/workflows`` canary that drives them.
"""

from __future__ import annotations
