"""Connector subsystem (#328).

Chat-platform layer on top of core. Core has no plugin concept; this
module manages connections, bindings, and runtimes as a peer Python
module. Module-boundary discipline: ``aios.*`` never imports from
``aios_connectors.*`` — the dependency arrow goes core → interface ←
subsystem, where the interfaces live in core (e.g.
``aios.tools.providers.ToolProvider``).
"""

from __future__ import annotations
