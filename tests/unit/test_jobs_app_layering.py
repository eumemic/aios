"""Layering guard for the job-queue infra split (issue #1476).

The job-queue *infrastructure* — the ``app`` singleton, the connector
builders, and the ``defer_*`` deferral primitives — lives in ``aios.jobs.app``,
a module BELOW both ``services`` and ``harness``. Importing it to *defer* a job
must NOT transitively pull in the harness execution graph (``loop``,
``completion``, ``trigger_runner``, the workflow step, …). That eager pull-in
was the whole cost this split removes: the API process defers jobs but never
runs a worker, yet used to pay the full harness import cost at startup.

Registration of ``@app.task`` handlers is decoupled: it happens only where the
tasks execute (the worker entrypoint imports ``aios.harness.tasks`` for its
side effect), never as a side effect of importing the app.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_jobs_app_exposes_infra_and_defer_primitives() -> None:
    from procrastinate import App

    from aios.jobs import app as jobs_app

    assert isinstance(jobs_app.app, App)
    # The three deferral primitives moved here from services.wake.
    assert callable(jobs_app.defer_wake)
    assert callable(jobs_app.defer_run_wake)
    assert callable(jobs_app.defer_trigger_fire)
    # The connector helpers moved here from harness.procrastinate_app.
    assert callable(jobs_app._build_connector)
    assert callable(jobs_app._sync_dsn)


def test_importing_jobs_app_does_not_pull_in_harness_execution_graph() -> None:
    """Importing the job-queue app (the defer path) must not import the
    harness execution graph. Run in a fresh interpreter so no other test's
    imports mask a transitive pull-in."""
    script = textwrap.dedent(
        """
        import sys
        import aios.jobs.app  # noqa: F401

        forbidden = [
            "aios.harness.tasks",
            "aios.harness.loop",
            "aios.harness.trigger_runner",
            "aios.harness.completion",
            "aios.workflows.step",
        ]
        leaked = [name for name in forbidden if name in sys.modules]
        if leaked:
            raise SystemExit("harness execution graph leaked into aios.jobs.app: " + repr(leaked))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_jobs_app_does_not_register_tasks() -> None:
    """Registration is a worker-only side effect. Importing the app in a fresh
    interpreter (without importing ``aios.harness.tasks``) registers none of the
    harness ``@app.task`` handlers — only procrastinate's own builtin tasks are
    present. Importing ``aios.harness.tasks`` then registers them."""
    script = textwrap.dedent(
        """
        from aios.jobs.app import app
        harness_before = {name for name in app.tasks if name.startswith("harness.")}
        if harness_before:
            raise SystemExit("harness tasks registered on import: " + repr(sorted(harness_before)))
        import aios.harness.tasks  # noqa: F401  — side-effect: register @app.task
        expected = {"harness.wake_session", "harness.run_trigger", "harness.wake_workflow"}
        missing = expected - set(app.tasks)
        if missing:
            raise SystemExit("tasks not registered after import: " + repr(sorted(missing)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_boot_backstop_fires_when_harness_tasks_missing() -> None:
    """The worker-boot registration backstop (#1476) must FIRE in the degraded
    state it exists to catch: a worker App constructed without the
    ``import aios.harness.tasks`` side effect.

    procrastinate's ``App.__init__`` unconditionally registers its own builtin
    ``remove_old_jobs`` task, so ``app.tasks`` is NEVER empty — a bare
    ``assert app.tasks`` truthiness check is VACUOUS: it passes in exactly the
    missing-import state it must reject. This proves the strengthened per-item
    guard (``_REQUIRED_HARNESS_TASKS - set(app.tasks)``) catches that gap.

    Run in a fresh interpreter, building a fresh ``App`` (the ``@app.task``
    decorators bind against the ``aios.jobs.app`` singleton, never this
    instance), so no other import can mask the missing registration."""
    script = textwrap.dedent(
        """
        from procrastinate import App

        from aios.harness.worker import _REQUIRED_HARNESS_TASKS
        from aios.jobs.app import _build_connector

        # A worker App that never had aios.harness.tasks registered against it.
        app = App(connector=_build_connector())

        # The bare truthiness check the strengthened assert REPLACED is vacuous:
        # procrastinate's builtin task keeps app.tasks non-empty, so the old
        # `assert app.tasks` would have passed in this exact degraded state.
        if not app.tasks:
            raise SystemExit("precondition failed: expected procrastinate builtin task present")

        # The strengthened per-item guard fires: the required harness tasks are
        # absent, so the missing-set is non-empty and the worker-boot assert raises.
        missing = _REQUIRED_HARNESS_TASKS - set(app.tasks)
        if not missing:
            raise SystemExit(
                "backstop did NOT fire: required harness tasks present without importing them"
            )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_harness_procrastinate_app_module_is_gone() -> None:
    """The conflated module is deleted (delete-don't-deprecate); no thin
    re-export shim survives."""
    import importlib.util

    assert importlib.util.find_spec("aios.harness.procrastinate_app") is None
