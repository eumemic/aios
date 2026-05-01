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
    store = await memory_service.create_store(
        harness._pool, name=store_name, description="x-session sync probe", metadata={}
    )
    await memory_service.create_memory(
        harness._pool,
        store_id=store.id,
        path="/seed.md",
        content="seed\n",
        actor=memory_service.ApiActor(),
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
    from aios.harness.loop import refresh_session_mount_state
    from aios.ids import make_id
    from aios.models.agents import ToolSpec
    from aios.services import agents as agents_service
    from aios.services import environments as environments_service

    if harness._env_id is None:
        env = await environments_service.create_environment(
            harness._pool, name=f"test-env-{make_id('env')[-8:]}"
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
    )
    session = await sessions_service.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=harness._env_id,
        title="memory test",
        metadata={},
        resources=resources or None,
    )
    await refresh_session_mount_state(harness._pool, session.id)
    return session


async def _start_session_with_store(harness: Harness, store_id: str, store_name: str) -> Any:
    """Backwards-compatible wrapper for the cross-session sync tests."""
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
        a = await _start_session_with_store(docker_harness, store_id, "xsync-read")
        b = await _start_session_with_store(docker_harness, store_id, "xsync-read")

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
        a = await _start_session_with_store(docker_harness, store_id, "xsync-race")
        b = await _start_session_with_store(docker_harness, store_id, "xsync-race")

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
        a = await _start_session_with_store(docker_harness, store_id, "xsync-edit-refresh")

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
        a = await _start_session_with_store(docker_harness, store_id, "xsync-fresh")

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
        store_id = await _attach_store(docker_harness, "xsync-api")
        a = await _start_session_with_store(docker_harness, store_id, "xsync-api")

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)

        # API creates a new memory while the session is up.
        await memory_service.create_memory(
            docker_harness._pool,
            store_id=store_id,
            path="/api_made.md",
            content="from-api",
            actor=memory_service.ApiActor(),
        )

        # Session's read tool sees it via the shared FS.
        result = await read_handler(a.id, {"path": "/mnt/memory/xsync-api/api_made.md"})
        assert "error" not in result, result
        assert "from-api" in result["content"]

    async def test_bash_visible_cross_session_but_no_version(self, docker_harness: Harness) -> None:
        """Documented v2 limitation: bash writes propagate via shared FS
        (so other sessions see them) but produce no memory_versions row."""
        from aios.db import queries
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "xsync-bash")
        a = await _start_session_with_store(docker_harness, store_id, "xsync-bash")
        b = await _start_session_with_store(docker_harness, store_id, "xsync-bash")

        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(a.id, pool=docker_harness._pool)
        await sandbox.get_or_provision(b.id, pool=docker_harness._pool)

        # A bash-writes a new file.
        a_bash = await bash_handler(
            a.id,
            {"command": "echo bashed-by-a > /mnt/memory/xsync-bash/from_bash.md"},
        )
        assert a_bash.get("exit_code", 0) == 0, a_bash

        # B's bash sees it (cross-session FS visibility).
        b_bash = await bash_handler(b.id, {"command": "cat /mnt/memory/xsync-bash/from_bash.md"})
        assert "bashed-by-a" in b_bash["stdout"]

        # But: no memory_versions row was created. Documented v2 gap.
        async with docker_harness._pool.acquire() as conn:
            versions = await queries.list_memory_versions(conn, store_id)
        paths = [v.path for v in versions]
        assert "/from_bash.md" not in paths


@needs_docker
class TestMountUpdateRecyclesContainer:
    """Issue #198: ``update_session(resources=...)`` on a session whose
    container is already up triggers a recycle on the next step so the
    new mount set takes effect."""

    async def test_attach_via_update_makes_mount_visible(self, docker_harness: Harness) -> None:
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
        )
        await sessions_service.update_session(
            docker_harness._pool,
            session.id,
            resources=[
                MemoryStoreResource(
                    type="memory_store", memory_store_id=store.id, access="read_write"
                ),
            ],
        )
        await refresh_session_mount_state(docker_harness._pool, session.id)

        after = await bash_handler(
            session.id, {"command": "ls -d /mnt/memory/attach-mount && echo OK"}
        )
        assert after.get("exit_code") == 0, after
        assert "OK" in after["stdout"]

    async def test_detach_via_update_drops_mount(self, docker_harness: Harness) -> None:
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

        await sessions_service.update_session(docker_harness._pool, session.id, resources=[])
        await refresh_session_mount_state(docker_harness._pool, session.id)

        after = await bash_handler(session.id, {"command": "ls /mnt/memory 2>&1; true"})
        assert "detach-mount" not in after["stdout"]

    async def test_idempotent_update_does_not_recycle(self, docker_harness: Harness) -> None:
        from aios.harness.loop import refresh_session_mount_state
        from aios.tools.bash import bash_handler

        store_id = await _attach_store(docker_harness, "idem-mount")
        session = await _start_session_with_store(docker_harness, store_id, "idem-mount")
        sandbox = runtime.require_sandbox_registry()
        await sandbox.get_or_provision(session.id, pool=docker_harness._pool)
        original_container_id = sandbox.peek(session.id).container_id  # type: ignore[union-attr]

        await sessions_service.update_session(
            docker_harness._pool,
            session.id,
            resources=[
                MemoryStoreResource(
                    type="memory_store", memory_store_id=store_id, access="read_write"
                ),
            ],
        )
        await refresh_session_mount_state(docker_harness._pool, session.id)

        cached = sandbox.peek(session.id)
        assert cached is not None
        assert cached.container_id == original_container_id

        result = await bash_handler(
            session.id, {"command": "ls -d /mnt/memory/idem-mount && echo OK"}
        )
        assert "OK" in result["stdout"], result
