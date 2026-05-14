"""Connector subsystem (#328).

Chat-platform layer on top of core. Core has no plugin concept; this
module manages connections, bindings, and runtimes as a peer Python
module.

Module-boundary discipline: ``aios.*`` keeps its top-level imports
free of ``aios_connectors.*``; the dependency arrow goes
core → interface ← subsystem, with the interfaces in core (e.g.
``aios.tools.providers.ToolProvider``). Two narrow function-local
imports cross the line during process startup or per-request work:

* ``aios.harness.worker.worker_main`` registers
  ``SubsystemToolProvider`` against the Protocol slot.
* ``aios.api.app.lifespan`` does the same on the API side so the
  ``/context`` endpoint reuses the worker's prelude path.
* ``aios.services.inbound.handle_inbound`` imports
  ``resolve_target_session`` inside the function body — the inbound
  handler is where the resolver naturally belongs and the
  function-scoped import keeps module import graphs core-only.

Anywhere else, the rule is hard: top-level imports stay clean.
"""

from __future__ import annotations
