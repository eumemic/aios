"""AIOS workflows: the durable, deterministic-Python runtime (the dual of an agent).

A *workflow* is an orchestrator whose core is deterministic code rather than a
model. A *run* is a durable execution instance: its state lives entirely in an
append-only journal, and each wake re-executes the author script from the top,
replaying memoized capability results until it reaches the next unresolved one
(replay-with-memo). The author script runs out-of-process (a credential-free
subprocess) so it can never reach the worker's master key or DB pool.

Block 1 (this layer): the durable runtime core — schema + journal (``db.queries.
workflows``), determinism (:mod:`aios.workflows.determinism`), the out-of-process
script host, the ``.send()`` driver, ``run_workflow_step``, and ``gate()``.
``agent()`` lands in Block 2.
"""
