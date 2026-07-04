"""Job-queue infrastructure — the layer below both ``services`` and ``harness``.

``aios.jobs`` owns the procrastinate ``App`` singleton and the deferral
primitives that lower layers use to *enqueue* jobs. It depends only on
``procrastinate`` + ``aios.config`` + ``aios.db.queries`` — never on
``services`` or ``harness``. Both ``services`` and ``harness`` import *down*
into this module, so the package graph itself states the rule: enqueuing a job
(infra) is decoupled from executing one (harness). See issue #1476.
"""
