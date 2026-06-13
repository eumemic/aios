"""E2E tests for the ``github_repository`` session resource.

Two layers of coverage:

- ``TestApi`` — DB + FastAPI ASGI transport, no real git clone. Exercises
  the wire surface: create-with-resource, list/get, token rotation, the
  aios-vs-CMA departure (add/remove via session update), the per-resource
  endpoint guarding against memory-store ids.
- ``TestRealClone`` — real ``git`` against ``octocat/Hello-World``. Skipped
  if ``git`` is not on PATH or the network is unreachable. Verifies the
  cache → working-tree pipeline and that the embedded-token URL is what
  ends up in ``.git/config``.

We deliberately don't spin up a Docker sandbox here. The ``--volume`` path
is exercised by the broader sandbox/registry tests; this file's job is to
prove the resource lifecycle works end-to-end at the service + HTTP
layer. A separate manual probe (documented in the plan) verifies the
in-container experience.
"""

from __future__ import annotations

import secrets
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from aios.models.agents import ToolSpec
from aios.models.github_repositories import GithubRepositoryResource, GithubRepositoryResourceEcho
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import github_repositories as github_service
from aios.services import sessions as sessions_service
from tests.helpers.connections import authed_client, wired_app

pytestmark = pytest.mark.docker

_OCTOCAT_REPO = "https://github.com/octocat/Hello-World"


def _uniq() -> str:
    return secrets.token_hex(4)


def _pat() -> str:
    """A syntactically-plausible token for tests that don't actually clone.

    The probe confirmed CMA validates only ``min_length=1`` at create
    time; a fake token here is fine for everything except real clone."""
    return f"ghp_test{_uniq()}{_uniq()}{_uniq()}"


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def crypto_box(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    return CryptoBox.from_base64(get_settings().vault_key.get_secret_value())


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


@pytest.fixture
async def env_and_agent(pool: Any) -> tuple[str, str]:
    """Create a fresh env + agent for one test.

    The agent has no tools — we never run the loop here, just exercise
    the resource lifecycle. Each test gets a unique pair so cross-test
    state doesn't leak.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    env = await environments_service.create_environment(
        pool, name=f"ghrepo-env-{_uniq()}", account_id=account_id
    )
    agent = await agents_service.create_agent(
        pool,
        name=f"ghrepo-agent-{_uniq()}",
        model="fake/test",
        system="github_repository test",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    return env.id, agent.id


# ─── service-layer flows ──────────────────────────────────────────────────


class TestServiceLayer:
    async def test_attach_round_trip(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        token = _pat()
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": token,
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )

        assert len(session.resources) == 1
        echo = session.resources[0]
        assert echo.type == "github_repository"
        assert echo.id.startswith("ghrepo_")
        assert echo.mount_path == "/workspace/repo"
        # Token never echoed.
        assert "authorization_token" not in echo.model_dump()

    async def test_token_decrypts_round_trip(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Service-internal: provisioner needs the raw token; verify the
        encrypt-on-create + decrypt-on-read path returns the original."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        original_token = _pat()
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": original_token,
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        echo = session.resources[0]
        assert isinstance(echo, GithubRepositoryResourceEcho)
        async with pool.acquire() as conn:
            recovered = await github_service.get_session_token(
                conn, crypto_box, session.id, echo.id, account_id=account_id
            )
        assert recovered == original_token

    async def test_token_rotation_changes_blob_and_bumps_updated_at(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        first_token = _pat()
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": first_token,
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        original_echo = session.resources[0]
        assert isinstance(original_echo, GithubRepositoryResourceEcho)
        new_token = _pat()
        rotated = await github_service.rotate_token(
            pool,
            crypto_box,
            session_id=session.id,
            resource_id=original_echo.id,
            new_token=new_token,
            account_id=account_id,
        )
        assert rotated.id == original_echo.id
        assert rotated.url == original_echo.url
        assert rotated.mount_path == original_echo.mount_path
        assert rotated.updated_at > original_echo.updated_at

        async with pool.acquire() as conn:
            recovered = await github_service.get_session_token(
                conn, crypto_box, session.id, original_echo.id, account_id=account_id
            )
        assert recovered == new_token
        assert recovered != first_token

    async def test_git_identity_round_trip(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Identity supplied at create-time persists through the DB and
        echoes back on read.  Rotation overwrites it."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                        "git_user_name": "Agent JN",
                        "git_user_email": "agent+jn@example.com",
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        echo = session.resources[0]
        assert isinstance(echo, GithubRepositoryResourceEcho)
        assert echo.git_user_name == "Agent JN"
        assert echo.git_user_email == "agent+jn@example.com"

        rotated = await github_service.rotate_token(
            pool,
            crypto_box,
            session_id=session.id,
            resource_id=echo.id,
            new_token=_pat(),
            identity=("Different Author", "other@example.com"),
            account_id=account_id,
        )
        assert rotated.git_user_name == "Different Author"
        assert rotated.git_user_email == "other@example.com"

    async def test_token_only_rotation_preserves_identity(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Rotating just the PAT (no identity payload) must NOT silently
        clear a previously configured ``git_user_name`` / ``git_user_email``.
        The router omits ``identity`` from the service call when the
        update body has no identity fields set.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                        "git_user_name": "Agent JN",
                        "git_user_email": "agent+jn@example.com",
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        echo = session.resources[0]
        assert isinstance(echo, GithubRepositoryResourceEcho)

        rotated = await github_service.rotate_token(
            pool,
            crypto_box,
            session_id=session.id,
            resource_id=echo.id,
            new_token=_pat(),
            account_id=account_id,
        )
        assert rotated.git_user_name == "Agent JN"
        assert rotated.git_user_email == "agent+jn@example.com"

    async def test_git_identity_optional_defaults_none(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Identity unset on create stays NULL through the DB and echoes
        back as ``None`` — pre-#207 v1 behaviour preserved."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        echo = session.resources[0]
        assert isinstance(echo, GithubRepositoryResourceEcho)
        assert echo.git_user_name is None
        assert echo.git_user_email is None

    async def test_full_list_replace_detaches_then_reattaches(
        self, pool: Any, crypto_box: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """The aios departure: PUT /sessions/{id} with ``resources=[]``
        detaches everything; with a new list, attaches that set fresh."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        env_id, agent_id = env_and_agent
        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        assert len(session.resources) == 1

        # Detach all.
        cleared = await sessions_service.update_session(
            pool, session.id, resources=[], crypto_box=crypto_box, account_id=account_id
        )
        assert cleared.resources == []

        # Attach a fresh one.
        re_attached = await sessions_service.update_session(
            pool,
            session.id,
            resources=[
                GithubRepositoryResource.model_validate(
                    {
                        "type": "github_repository",
                        "url": "https://github.com/octocat/Spoon-Knife",
                        "mount_path": "/workspace/spoon",
                        "authorization_token": _pat(),
                    }
                )
            ],
            crypto_box=crypto_box,
            account_id=account_id,
        )
        assert len(re_attached.resources) == 1
        assert re_attached.resources[0].mount_path == "/workspace/spoon"


# ─── HTTP wire surface ─────────────────────────────────────────────────────


async def _create_env_and_agent_via_api(client: httpx.AsyncClient) -> tuple[str, str]:
    r = await client.post("/v1/environments", json={"name": f"ghrepo-env-{_uniq()}"})
    assert r.status_code == 201, r.text
    env_id = r.json()["id"]
    r = await client.post(
        "/v1/agents",
        json={
            "name": f"ghrepo-agent-{_uniq()}",
            "model": "fake/test",
            "system": "ghrepo http test",
            "tools": [{"type": "bash"}],
            "metadata": {},
            "window_min": 50_000,
            "window_max": 150_000,
        },
    )
    assert r.status_code == 201, r.text
    return env_id, r.json()["id"]


class TestApi:
    async def test_create_session_with_github_repo(self, http_client: httpx.AsyncClient) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                ],
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert len(body["resources"]) == 1
        echo = body["resources"][0]
        assert echo["type"] == "github_repository"
        assert echo["id"].startswith("ghrepo_")
        assert echo["url"] == _OCTOCAT_REPO
        assert echo["mount_path"] == "/workspace/repo"
        assert "authorization_token" not in echo

    async def test_token_required(self, http_client: httpx.AsyncClient) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                    }
                ],
            },
        )
        assert r.status_code == 422, r.text

    async def test_empty_token_rejected(self, http_client: httpx.AsyncClient) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": "",
                    }
                ],
            },
        )
        assert r.status_code == 422, r.text

    async def test_list_resources_endpoint(self, http_client: httpx.AsyncClient) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                ],
            },
        )
        assert r.status_code == 201, r.text
        sid = r.json()["id"]

        r = await http_client.get(f"/v1/sessions/{sid}/resources")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["type"] == "github_repository"

    async def test_get_single_resource(self, http_client: httpx.AsyncClient) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                ],
            },
        )
        sid = r.json()["id"]
        rid = r.json()["resources"][0]["id"]

        r = await http_client.get(f"/v1/sessions/{sid}/resources/{rid}")
        assert r.status_code == 200, r.text
        echo = r.json()
        assert echo["id"] == rid
        assert echo["type"] == "github_repository"
        assert "authorization_token" not in echo

    async def test_token_rotation_via_put(self, http_client: httpx.AsyncClient) -> None:
        # Rotation moved from POST to PUT /resources/{rid} (#270) so the
        # collection POST is free for the additive add-one operation.
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                ],
            },
        )
        sid = r.json()["id"]
        original_echo = r.json()["resources"][0]

        r = await http_client.put(
            f"/v1/sessions/{sid}/resources/{original_echo['id']}",
            json={"authorization_token": _pat()},
        )
        assert r.status_code == 200, r.text
        rotated = r.json()
        assert rotated["id"] == original_echo["id"]
        assert rotated["url"] == original_echo["url"]
        assert rotated["updated_at"] > original_echo["updated_at"]
        assert "authorization_token" not in rotated

    async def test_session_update_replaces_resource_list(
        self, http_client: httpx.AsyncClient
    ) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "metadata": {},
                "resources": [
                    {
                        "type": "github_repository",
                        "url": _OCTOCAT_REPO,
                        "mount_path": "/workspace/repo",
                        "authorization_token": _pat(),
                    }
                ],
            },
        )
        sid = r.json()["id"]

        # Detach.
        r = await http_client.put(f"/v1/sessions/{sid}", json={"resources": []})
        assert r.status_code == 200, r.text
        assert r.json()["resources"] == []

        # Re-attach.
        r = await http_client.put(
            f"/v1/sessions/{sid}",
            json={
                "resources": [
                    {
                        "type": "github_repository",
                        "url": "https://github.com/octocat/Spoon-Knife",
                        "mount_path": "/workspace/spoon",
                        "authorization_token": _pat(),
                    }
                ]
            },
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["resources"]) == 1
        assert r.json()["resources"][0]["mount_path"] == "/workspace/spoon"

    async def test_memstore_id_rejected_on_per_resource_endpoints(
        self, http_client: httpx.AsyncClient
    ) -> None:
        env_id, agent_id = await _create_env_and_agent_via_api(http_client)
        r = await http_client.post(
            "/v1/sessions",
            json={"agent_id": agent_id, "environment_id": env_id, "metadata": {}},
        )
        sid = r.json()["id"]
        # Use a memory store id shape with a valid 26-char Crockford-base32
        # ULID body so it passes the malformed-id guard and falls through
        # to the prefix-mismatch branch in ``_require_github_resource_id``.
        # Both branches return ValidationError (422), but the message
        # distinguishes them — assert on the prefix-mismatch text.
        r = await http_client.get(
            f"/v1/sessions/{sid}/resources/memstore_01H0RKEMFTPM3J3Q3WCABCDEFG"
        )
        assert r.status_code == 422, r.text
        assert "ghrepo_" in r.json()["error"]["message"]


# ─── real-network clone (octocat/Hello-World) ─────────────────────────────


def _git_available() -> bool:
    return shutil.which("git") is not None


def _network_available() -> bool:
    """Cheap connectivity check — just resolve github.com."""
    import socket

    try:
        socket.gethostbyname("github.com")
        return True
    except OSError:
        return False


needs_real_clone = pytest.mark.skipif(
    not (_git_available() and _network_available()),
    reason="needs git on PATH and reachable github.com",
)


@needs_real_clone
class TestRealClone:
    async def test_cache_clone_then_session_clone_origin_scrubbed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end clone of a tiny public repo. After
        ``ensure_session_working_tree``, ``.git/config`` must hold the
        proxy URL — never the auth-embedded one. This regression-tests
        issue #208.

        The PAT for octocat/Hello-World needs only ``public_repo`` — but
        for unauthenticated cloning of a public repo, GitHub rejects
        bogus PATs. We patch ``_build_auth_url`` to a no-op so the
        clone is anonymous, then verify the post-clone scrub still
        replaces ``origin`` with the proxy URL.
        """
        from aios.config import get_settings

        s = get_settings()
        monkeypatch.setattr(s, "workspace_root", tmp_path)

        from aios.sandbox import github_clone

        monkeypatch.setattr(github_clone, "_build_auth_url", lambda url, _tok: url)

        cache_dir = await github_clone.ensure_cache_clone(_OCTOCAT_REPO, "ignored")
        assert cache_dir.exists()
        assert (cache_dir / "HEAD").exists()
        assert (cache_dir / "objects").is_dir()

        proxy_url = "http://aios.test:9999/git/SECRET/octocat/Hello-World"
        work_dir = await github_clone.ensure_session_working_tree(
            session_id="sess_test",
            resource_id="ghrepo_test",
            repo_url=_OCTOCAT_REPO,
            token="REAL_TOKEN_ghp_xxxx",
            cache_dir=cache_dir,
            proxy_url=proxy_url,
        )
        assert work_dir.exists()
        assert (work_dir / "README").exists()
        assert (work_dir / ".git" / "config").exists()

        config_text = (work_dir / ".git" / "config").read_text()
        # And it MUST carry the proxy URL.
        assert proxy_url in config_text

        # Scan the entire .git tree (config, refs, logs/HEAD reflog,
        # FETCH_HEAD, ORIG_HEAD, packed-refs, etc.) to catch any path
        # where the token might leak — issue #208 was an instance of
        # exactly this kind of "we only checked the obvious file" bug.
        for path in (work_dir / ".git").rglob("*"):
            if not path.is_file():
                continue
            try:
                contents = path.read_bytes()
            except OSError:
                continue
            assert b"REAL_TOKEN_ghp_xxxx" not in contents, (
                f"token leaked into {path.relative_to(work_dir)}"
            )
            assert b"x-access-token" not in contents, (
                f"x-access-token URL form leaked into {path.relative_to(work_dir)}"
            )

    async def test_per_session_clone_recreated_on_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling ``ensure_session_working_tree`` twice rms-and-re-clones
        — that's how token rotation propagates the new auth into
        ``.git/config`` without any "is this stale?" detection logic."""
        from aios.config import get_settings

        s = get_settings()
        monkeypatch.setattr(s, "workspace_root", tmp_path)

        from aios.sandbox import github_clone

        monkeypatch.setattr(github_clone, "_build_auth_url", lambda url, _tok: url)

        cache_dir = await github_clone.ensure_cache_clone(_OCTOCAT_REPO, "ignored")
        proxy_url = "http://aios.test:9999/git/SECRET/octocat/Hello-World"

        work_dir = await github_clone.ensure_session_working_tree(
            session_id="sess_a",
            resource_id="ghrepo_a",
            repo_url=_OCTOCAT_REPO,
            token="t1",
            cache_dir=cache_dir,
            proxy_url=proxy_url,
        )
        # Drop a sentinel that recreate must clobber.
        (work_dir / "SENTINEL").write_text("sentinel\n")
        assert (work_dir / "SENTINEL").exists()

        await github_clone.ensure_session_working_tree(
            session_id="sess_a",
            resource_id="ghrepo_a",
            repo_url=_OCTOCAT_REPO,
            token="t2",
            cache_dir=cache_dir,
            proxy_url=proxy_url,
        )
        assert not (work_dir / "SENTINEL").exists()
        # Working tree is fresh (README still present, sentinel gone).
        assert (work_dir / "README").exists()
