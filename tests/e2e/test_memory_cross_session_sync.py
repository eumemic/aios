"""E2E for cross-session memory sync (v2).

Two sessions attached read_write to the same store. Real Postgres + real
Docker bind-mount so the v2 shared-host-dir + sha-precondition architecture
is exercised end-to-end. Drives tool handlers directly rather than the full
agent loop — keeps the test focused on the sync semantics.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.models.memory_stores import MemoryStoreResource
from aios.services import memory_stores as memory_service
from aios.services import sessions as sessions_service
from aios.tools.edit import edit_handler
from aios.tools.read import read_handler
from aios.tools.write import write_handler
from tests.conftest import needs_docker
from tests.e2e.harness import Harness


async def _attach_store(harness: Harness, store_name: str) -> str:
    """Create a memory store and seed /seed.md. Returns the store id."""
    account_id = "acc_test_stub"  # PR 3 scaffolding
    store = await memory_service.create_store(
        harness._pool,
        name=store_name,
        description="x-session sync probe",
        metadata={},
        account_id=account_id,
    )
    await memory_service.create_memory(
        harness._pool,
        store_id=store.id,
        path="/seed.md",
        content="seed\n",
        actor=memory_service.ApiActor(),
        account_id=account_id,
    )
    return store.id


async def _start_session(
    harness: Harness,
    *,
    resources: list[MemoryStoreResource] | None = None,
    tools: tuple[str, ...] = ("read", "write", "bash"),
) -> Any:
    """Create env+agent+session; prime the runtime mount cache via the
    same helper the worker step body uses, so tests follow the production
    path for mount visibility.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.harness.loop import refresh_session_mount_state
    from aios.ids import make_id
    from aios.models.agents import ToolSpec
    from aios.services import agents as agents_service
    from aios.services import environments as environments_service

    if harness._env_id is None:
        env = await environments_service.create_environment(
            harness._pool, name=f"test-env-{make_id('env')[-8:]}", account_id=account_id
        )
        harness._env_id = env.id

    agent = await agents_service.create_agent(
        harness._pool,
        name=f"test-agent-{make_id('agent')[-8:]}",
        model="fake/test",
        system="memory test",
        tools=[ToolSpec(type=t) for t in tools],  # type: ignore[arg-type]
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    session = await sessions_service.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=harness._env_id,
        title="memory test",
        metadata={},
        resources=resources,
        account_id=account_id,
    )
    await refresh_session_mount_state(harness._pool, session.id)
    return session


async def _start_session_with_store(harness: Harness, store_id: str) -> Any:
    """Convenience: ``_start_session`` attached read_write to one store."""
    return await _start_session(
        harness,
        resources=[
            MemoryStoreResource(type="memory_store", memory_store_id=store_id, access="read_write"),
        ],
    )


@needs_docker
class TestCrossSessionSync:
    async def test_write_in_a_visible_to_b_via_read(self, docker_harness: Harness) -> None:
        """A's write tool propagates to B's read tool through the shared FS."""
        store_id = await _attach_store(docker_harness, "xsync-read")
        a = await _start_session_with_store(docker_harness, store_id)
        b = await _start_session_with_store(docker_harness, store_id)

        # Provisioning both containers materializes the shared dir once.
        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)
        await sandbox.get_or_provision(b.id, pool=docker_harness._pool)

        # A writes via tool.
        result = await write_handler(
            a.id, {"path": "/mnt/memory/xsync-read/shared.md", "content": "from-A"}
        )
        assert "error" not in result, result

        # B's read tool sees A's content.
        b_result = await read_handler(b.id, {"path": "/mnt/memory/xsync-read/shared.md"})
        assert "error" not in b_result, b_result
        # cat -n format: "     1\tfrom-A"
        assert "from-A" in b_result["content"]

    async def test_concurrent_writes_use_precondition(self, docker_harness: Harness) -> None:
        """Both sessions read /shared.md (caching same sha). A writes →
        commits with the cached sha. B writes → DB sha has moved on, B's
        write fails with the typed precondition error."""
        store_id = await _attach_store(docker_harness, "xsync-race")
        a = await _start_session_with_store(docker_harness, store_id)
        b = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)
        await sandbox.get_or_provision(b.id, pool=docker_harness._pool)

        # Both sessions read /seed.md — both cache the seed sha.
        a_read = await read_handler(a.id, {"path": "/mnt/memory/xsync-race/seed.md"})
        b_read = await read_handler(b.id, {"path": "/mnt/memory/xsync-race/seed.md"})
        assert "error" not in a_read
        assert "error" not in b_read

        # A writes successfully (sha matches).
        a_write = await write_handler(
            a.id,
            {"path": "/mnt/memory/xsync-race/seed.md", "content": "from-A\n"},
        )
        assert "error" not in a_write, a_write

        # B writes with stale cached sha → precondition failure.
        b_write = await write_handler(
            b.id,
            {"path": "/mnt/memory/xsync-race/seed.md", "content": "from-B\n"},
        )
        assert "error" in b_write
        assert "changed since your last read" in b_write["error"]

        # B re-reads (refreshes cached sha to current DB sha) and retries.
        await read_handler(b.id, {"path": "/mnt/memory/xsync-race/seed.md"})
        b_retry = await write_handler(
            b.id,
            {"path": "/mnt/memory/xsync-race/seed.md", "content": "from-B\n"},
        )
        assert "error" not in b_retry, b_retry

    async def test_edit_refreshes_read_sha_for_subsequent_write(
        self, docker_harness: Harness
    ) -> None:
        """After read -> edit, a subsequent write tool call against the same
        path must succeed: the edit should have refreshed the cached read-sha
        so the write's precondition matches the post-edit DB state."""
        store_id = await _attach_store(docker_harness, "xsync-edit-refresh")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # Read primes the cache with the seed sha.
        r = await read_handler(a.id, {"path": "/mnt/memory/xsync-edit-refresh/seed.md"})
        assert "error" not in r, r

        # Edit changes the DB content; without the cache refresh fix the next
        # write would 409 with a stale precondition.
        e = await edit_handler(
            a.id,
            {
                "path": "/mnt/memory/xsync-edit-refresh/seed.md",
                "old_string": "seed",
                "new_string": "edited",
            },
        )
        assert "error" not in e, e

        w = await write_handler(
            a.id,
            {
                "path": "/mnt/memory/xsync-edit-refresh/seed.md",
                "content": "rewritten\n",
            },
        )
        assert "error" not in w, w

    async def test_fresh_write_no_precondition(self, docker_harness: Harness) -> None:
        """Writing to a path the model never read — no precondition; succeeds."""
        store_id = await _attach_store(docker_harness, "xsync-fresh")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        result = await write_handler(
            a.id,
            {"path": "/mnt/memory/xsync-fresh/never_read.md", "content": "fresh"},
        )
        assert "error" not in result, result

    async def test_api_write_visible_to_attached_session(self, docker_harness: Harness) -> None:
        """API-driven create_memory mirrors to host dir; attached session's
        read tool sees the new content."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        store_id = await _attach_store(docker_harness, "xsync-api")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # API creates a new memory while the session is up.
        await memory_service.create_memory(
            docker_harness._pool,
            store_id=store_id,
            path="/api_made.md",
            content="from-api",
            actor=memory_service.ApiActor(),
            account_id=account_id,
        )

        # Session's read tool sees it via the shared FS.
        result = await read_handler(a.id, {"path": "/mnt/memory/xsync-api/api_made.md"})
        assert "error" not in result, result
        assert "from-api" in result["content"]

    async def test_bash_create_reconciles_to_version(self, docker_harness: Harness) -> None:
        """bash writes to memory mounts are reconciled into memory_versions by the post-exec hook."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "xsync-bash")
        a = await _start_session_with_store(docker_harness, store_id)
        b = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)
        await sandbox.get_or_provision(b.id, pool=docker_harness._pool)

        # A bash-writes a new file; reconcile hook fires after exec.
        a_bash = await bash_handler(
            a.id,
            {"command": "echo bashed-by-a > /mnt/memory/xsync-bash/from_bash.md"},
        )
        assert a_bash.get("exit_code", 0) == 0, a_bash

        # B's bash sees it (cross-session FS visibility).
        b_bash = await bash_handler(b.id, {"command": "cat /mnt/memory/xsync-bash/from_bash.md"})
        assert "bashed-by-a" in b_bash["stdout"]

        # Post-exec reconcile created a memory_versions row.
        async with docker_harness._pool.acquire() as conn:
            versions = await queries.list_memory_versions(conn, store_id, account_id=account_id)
        paths = [v.path for v in versions]
        assert "/from_bash.md" in paths

        # Version was stamped as session_actor.
        version = next(v for v in versions if v.path == "/from_bash.md")
        assert version.created_by.type == "session_actor"

    async def test_bash_modify_reconciles_version(self, docker_harness: Harness) -> None:
        """bash modifies existing file; a 'modified' version row appears."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "xsync-bash-mod")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # Modify the pre-seeded /seed.md via bash.
        # snapshot before exec sees seed.md; after exec sees changed content.
        result = await bash_handler(
            a.id,
            {"command": "echo modified-by-bash > /mnt/memory/xsync-bash-mod/seed.md"},
        )
        assert result.get("exit_code", 0) == 0, result

        async with docker_harness._pool.acquire() as conn:
            versions = await queries.list_memory_versions(conn, store_id, account_id=account_id)
        # There should be at least 2 versions: created (from seed) + modified (from bash).
        seed_versions = [v for v in versions if v.path == "/seed.md"]
        operations = {v.operation for v in seed_versions}
        assert "modified" in operations

    async def test_bash_delete_reconciles_version(self, docker_harness: Harness) -> None:
        """bash rm's a file; a 'deleted' version row appears."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "xsync-bash-del")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # Delete /seed.md via bash.
        result = await bash_handler(
            a.id,
            {"command": "rm /mnt/memory/xsync-bash-del/seed.md"},
        )
        assert result.get("exit_code", 0) == 0, result

        async with docker_harness._pool.acquire() as conn:
            versions = await queries.list_memory_versions(conn, store_id, account_id=account_id)
        seed_versions = [v for v in versions if v.path == "/seed.md"]
        operations = {v.operation for v in seed_versions}
        assert "deleted" in operations

    async def test_bash_binary_file_reconcile_warning(self, docker_harness: Harness) -> None:
        """bash writes binary bytes; stderr contains reconcile warning; no version row created."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "xsync-bash-bin")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # Write binary (non-UTF-8) bytes to a memory mount file.
        result = await bash_handler(
            a.id,
            {"command": "printf '\\xff\\xfe\\x00\\x01' > /mnt/memory/xsync-bash-bin/binary.bin"},
        )
        assert result.get("exit_code", 0) == 0, result

        # Warning should appear in stderr.
        assert "[memory-reconcile]" in result["stderr"]

        # No version row created for the binary file.
        async with docker_harness._pool.acquire() as conn:
            versions = await queries.list_memory_versions(conn, store_id, account_id=account_id)
        paths = [v.path for v in versions]
        assert "/binary.bin" not in paths

    async def test_bash_nohup_residual_gap(self, docker_harness: Harness) -> None:
        """Background processes after bash returns may not be captured.

        Documents that nohup/background bash writes to memory mounts that
        complete *after* the command returns fall outside the reconcile window.
        The test simply asserts no crash occurs — not that the write is captured.
        """
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "xsync-bash-nohup")
        a = await _start_session_with_store(docker_harness, store_id)

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # Background process that writes after bash -c returns.
        result = await bash_handler(
            a.id,
            {
                "command": "nohup sh -c 'sleep 0.5; echo done > /mnt/memory/xsync-bash-nohup/bg.md' & echo done"
            },
        )
        # No crash; exit_code check is lenient — the command itself should succeed.
        assert result.get("exit_code", 0) == 0, result


@needs_docker
class TestMountUpdateRecyclesContainer:
    """Issue #198: ``update_session(resources=...)`` on a session whose
    container is already up triggers a recycle on the next step so the
    new mount set takes effect."""

    async def test_attach_via_update_makes_mount_visible(self, docker_harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.harness.loop import refresh_session_mount_state
        from aios.tools.bash import bash_handler

        session = await _start_session(docker_harness, tools=("bash",))

        # Provision the container BEFORE attaching the store — this is the
        # critical case the recycle logic exists for.
        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(session.id, pool=docker_harness._pool)

        before = await bash_handler(session.id, {"command": "ls /mnt/memory 2>&1; true"})
        assert "attach-mount" not in before["stdout"]

        store = await memory_service.create_store(
            docker_harness._pool,
            name="attach-mount",
            description="attach probe",
            metadata={},
            account_id=account_id,
        )
        await sessions_service.update_session(
            docker_harness._pool,
            session.id,
            resources=[
                MemoryStoreResource(
                    type="memory_store", memory_store_id=store.id, access="read_write"
                ),
            ],
            account_id=account_id,
        )
        await refresh_session_mount_state(docker_harness._pool, session.id)

        after = await bash_handler(
            session.id, {"command": "ls -d /mnt/memory/attach-mount && echo OK"}
        )
        assert after.get("exit_code") == 0, after
        assert "OK" in after["stdout"]

    async def test_detach_via_update_drops_mount(self, docker_harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.harness.loop import refresh_session_mount_state
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "detach-mount")
        session = await _start_session(
            docker_harness,
            resources=[
                MemoryStoreResource(
                    type="memory_store", memory_store_id=store_id, access="read_write"
                ),
            ],
            tools=("bash",),
        )
        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(session.id, pool=docker_harness._pool)
        before = await bash_handler(
            session.id, {"command": "ls -d /mnt/memory/detach-mount && echo OK"}
        )
        assert "OK" in before["stdout"], before

        await sessions_service.update_session(
            docker_harness._pool, session.id, resources=[], account_id=account_id
        )
        await refresh_session_mount_state(docker_harness._pool, session.id)

        after = await bash_handler(session.id, {"command": "ls /mnt/memory 2>&1; true"})
        assert "detach-mount" not in after["stdout"]

    async def test_idempotent_update_does_not_recycle(self, docker_harness: Harness) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.harness.loop import refresh_session_mount_state
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "idem-mount")
        session = await _start_session_with_store(docker_harness, store_id)
        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(session.id, pool=docker_harness._pool)
        original_container_id = sandbox.peek(session.id).sandbox_id  # type: ignore[union-attr]

        await sessions_service.update_session(
            docker_harness._pool,
            session.id,
            resources=[
                MemoryStoreResource(
                    type="memory_store", memory_store_id=store_id, access="read_write"
                ),
            ],
            account_id=account_id,
        )
        await refresh_session_mount_state(docker_harness._pool, session.id)

        cached = sandbox.peek(session.id)
        assert cached is not None
        assert cached.sandbox_id == original_container_id

        result = await bash_handler(
            session.id, {"command": "ls -d /mnt/memory/idem-mount && echo OK"}
        )
        assert "OK" in result["stdout"], result
