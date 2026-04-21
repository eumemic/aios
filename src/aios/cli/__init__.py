"""Operator-facing CLI subcommands (progresses #35 item 4).

Each submodule implements one top-level verb group (``connections``,
eventually ``vaults``, ``routing-rules``, etc.).  Dispatched from
:mod:`aios.__main__`.
"""

from __future__ import annotations
