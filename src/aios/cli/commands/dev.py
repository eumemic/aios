"""``aios dev ...`` — per-worktree dev instance management.

Lets each git worktree run its own isolated aios against a shared local
Postgres. ``bootstrap`` creates a per-worktree database, picks a free port,
writes a ``.env`` with instance-specific overrides, and runs migrations.
``teardown`` drops the database and prunes per-instance containers.
``status`` reports instance identity and liveness.

The commands parse ``.env`` directly (rather than going through
:class:`aios.config.Settings`) so destructive cleanup still works when the
DB is unreachable or a secrets file is malformed.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse, urlunparse

import asyncpg
import typer

from aios.cli.output import print_error, print_note

# ── instance id derivation + validation ────────────────────────────────────

INSTANCE_ID_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")

# PostgreSQL identifiers are silently truncated at 63 bytes. We prefix
# dev databases with "aios_dev_" (9 bytes), leaving 54 for the instance id.
DB_PREFIX = "aios_dev_"
MAX_DB_NAME_BYTES = 63
MAX_INSTANCE_ID_LEN = MAX_DB_NAME_BYTES - len(DB_PREFIX)


def derive_instance_id(repo_root: Path) -> str:
    """Derive a collision-free instance id from a git repo root.

    The id is always ``<sanitized-basename>_<8-char-hash>`` so that two
    worktrees with the same basename on different paths never collide.
    Hash input is the fully resolved path.

    Output is guaranteed to match :data:`INSTANCE_ID_PATTERN` and fit
    inside ``aios_dev_<id>`` at ≤63 bytes.
    """
    resolved = str(repo_root.resolve())
    raw = repo_root.name.lower()
    safe = re.sub(r"[^a-z0-9_]", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    suffix = hashlib.sha256(resolved.encode()).hexdigest()[:8]
    if not safe or safe[0].isdigit():
        safe = "w_" + safe
    # 1 underscore + 8-char suffix takes 9 bytes.
    max_prefix = MAX_INSTANCE_ID_LEN - 1 - 8
    return f"{safe[:max_prefix]}_{suffix}"


def validate_instance_id_for_db(instance_id: str) -> None:
    """Validate an instance id before it flows into SQL or a docker label.

    Checks the safe-char pattern and the 63-byte Postgres-identifier limit
    (PG silently truncates beyond that, which would silently create the
    wrong database).
    """
    if not INSTANCE_ID_PATTERN.fullmatch(instance_id):
        raise ValueError(
            f"invalid AIOS_INSTANCE_ID {instance_id!r}: must match {INSTANCE_ID_PATTERN.pattern}"
        )
    full = DB_PREFIX + instance_id
    if len(full.encode()) > MAX_DB_NAME_BYTES:
        raise ValueError(
            f"AIOS_INSTANCE_ID {instance_id!r} produces DB name {full!r} "
            f"({len(full)} bytes); Postgres truncates identifiers over "
            f"{MAX_DB_NAME_BYTES} bytes"
        )


# ── env-file I/O ───────────────────────────────────────────────────────────

_ENV_LINE_RE = re.compile(r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv-style file into a plain dict.

    Ignores blank lines and ``#`` comments. Strips one layer of surrounding
    single or double quotes from values. Does NOT expand ``$VAR``
    references or perform any merging with process env — that's the caller's
    choice. Missing files return ``{}``.
    """
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(line)
        if not m:
            continue
        key = m.group("key")
        value = m.group("value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result


def write_env_atomic(path: Path, updates: dict[str, str]) -> None:
    """Merge ``updates`` into ``path`` and rewrite atomically.

    Preserves unrelated keys and line ordering: existing lines whose keys
    appear in ``updates`` are rewritten in place, and remaining ``updates``
    keys are appended at the end. Comments and blank lines pass through.
    Write goes to a tmpfile in the same directory then ``os.replace`` —
    avoids ever leaving a half-written ``.env``.
    """
    existing_text = path.read_text() if path.exists() else ""
    remaining = dict(updates)
    new_lines: list[str] = []
    for raw_line in existing_text.splitlines():
        m = _ENV_LINE_RE.match(raw_line)
        if m and m.group("key") in remaining:
            key = m.group("key")
            new_lines.append(f"{key}={remaining.pop(key)}")
        else:
            new_lines.append(raw_line)
    for key, value in remaining.items():
        new_lines.append(f"{key}={value}")
    new_text = "\n".join(new_lines) + ("\n" if new_lines else "")

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(new_text)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_instance_env(repo_root: Path) -> dict[str, str]:
    """Read the worktree's ``.env`` and require the four instance keys.

    Used by teardown/status — both need to work when ``Settings`` would
    fail to load (e.g. broken secrets file, DB unreachable). Raises
    ``typer.Exit(1)`` with a clear message if any required key is missing.
    """
    env_path = repo_root / ".env"
    env = parse_env_file(env_path)
    required = (
        "AIOS_INSTANCE_ID",
        "AIOS_DB_URL",
        "AIOS_API_PORT",
        "AIOS_WORKSPACE_ROOT",
    )
    missing = [k for k in required if k not in env]
    if missing:
        print_error(
            f"{env_path} is missing required keys: {', '.join(missing)}. "
            f"Run `aios dev bootstrap` first."
        )
        raise typer.Exit(1)
    return env


# ── URL + port helpers ─────────────────────────────────────────────────────


def derive_runtime_db_url(admin_url: str, instance_id: str) -> str:
    """Return an AIOS_DB_URL for this instance by swapping the DB name only.

    Preserves scheme, host, port, user, password, and query from the admin
    URL. This keeps bootstrap working against non-default Postgres setups
    (alternate ports, remote hosts, alternate credentials).
    """
    parsed = urlparse(admin_url)
    db_name = DB_PREFIX + instance_id
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def pick_free_port() -> int:
    """Bind to port 0 on loopback, return the kernel-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    """Best-effort check: does something accept on ``host:port``?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def get_git_repo_root() -> Path:
    """Return the current git repository top level, or raise typer.Exit(1)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("not inside a git repository (git rev-parse failed)")
        raise typer.Exit(1) from None
    return Path(out.strip())


# ── shared-DB-from-worktree guard ──────────────────────────────────────────


def is_linked_worktree() -> bool:
    """Return True iff the cwd is inside a *linked* git worktree.

    A linked worktree (created by ``git worktree add``) has its ``--git-dir``
    point at ``<main>/.git/worktrees/<name>`` while ``--git-common-dir`` points
    at the main checkout's ``.git``; they differ. In the main checkout both
    resolve to the same path (``.git`` as-returned) so they are equal.

    We compare both values *as returned* (NOT ``--absolute-git-dir``): in the
    main checkout both are the relative string ``.git`` and compare equal,
    which is exactly what we want.

    Swallows ``CalledProcessError`` / ``FileNotFoundError`` → ``False`` so it
    is prod-safe: the api image has no git binary, and the worker's ``/app``
    has no ``.git`` because ``.dockerignore`` excludes it. Both cases must
    never raise — they just mean "not a linked worktree."
    """
    try:
        git_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return git_dir != common_dir


def db_name_is_shared(db_url: str) -> bool:
    """Return True iff ``db_url``'s database name is NOT a bootstrapped one.

    Every database created by ``aios dev bootstrap`` is named
    ``aios_dev_<id>`` (see :func:`derive_runtime_db_url`). Anything else —
    the shared local dev DB (``aios``), test DBs, etc. — is "shared" for the
    purposes of the worktree guard.
    """
    db_name = urlparse(db_url).path.lstrip("/")
    return not db_name.startswith(DB_PREFIX)


def runtime_db_url(repo_root: Path) -> str | None:
    """Resolve ``AIOS_DB_URL`` the way pydantic will at runtime.

    Process env beats the worktree ``.env`` (``config.py`` env_file tuple),
    so honour an exported value first, then fall back to the worktree
    ``.env``. Returns ``None`` if neither sets it. Avoids constructing the
    full :class:`Settings` so the guard never depends on a loadable config.
    """
    exported = os.environ.get("AIOS_DB_URL")
    if exported:
        return exported
    env = parse_env_file(repo_root / ".env")
    return env.get("AIOS_DB_URL")


def assert_not_shared_in_worktree() -> None:
    """Abort startup if a linked worktree is about to use the shared dev DB.

    The footgun (#349): in a git worktree you source the main checkout's
    ``.env`` (or never ran ``aios dev bootstrap``), so the runtime
    ``AIOS_DB_URL`` points at the shared local dev DB while you believe you're
    isolated — silently writing real records into a shared/prod-shadow DB.

    Guard condition: running from a *linked worktree* AND the resolved runtime
    ``AIOS_DB_URL`` db-name is not ``aios_dev_*``. ``AIOS_ALLOW_SHARED_DB=1``
    is an explicit escape hatch.
    """
    if os.environ.get("AIOS_ALLOW_SHARED_DB") == "1":
        return
    if not is_linked_worktree():
        return
    db_url = runtime_db_url(get_git_repo_root())
    if db_url is None or not db_name_is_shared(db_url):
        return
    print_error(
        "refusing to start: this linked git worktree is pointing at the "
        "shared dev DB (AIOS_DB_URL does not name an aios_dev_* database). "
        "You probably sourced the main checkout's .env. Run `aios dev "
        "bootstrap` and source the worktree-local .env in a fresh shell, or "
        "set AIOS_ALLOW_SHARED_DB=1 to override."
    )
    raise typer.Exit(1)


def classify_instance_mode(
    env: dict[str, str],
    resolved_db_url: str | None,
    in_linked_worktree: bool,
) -> str:
    """Classify the worktree's dev mode for ``aios dev status``.

    Returns one of ``"unbootstrapped"``, ``"shared"``, ``"isolated"``:

    - ``unbootstrapped`` — the worktree ``.env`` lacks the four instance keys
      (``aios dev bootstrap`` was never run here).
    - ``shared`` — a linked worktree whose resolved runtime DB is shared.
    - ``isolated`` — everything else (main checkout, or a bootstrapped
      worktree pointing at its ``aios_dev_*`` DB).
    """
    required = (
        "AIOS_INSTANCE_ID",
        "AIOS_DB_URL",
        "AIOS_API_PORT",
        "AIOS_WORKSPACE_ROOT",
    )
    if any(k not in env for k in required):
        return "unbootstrapped"
    if in_linked_worktree and resolved_db_url is not None and db_name_is_shared(resolved_db_url):
        return "shared"
    return "isolated"


# ── stale-export guard ─────────────────────────────────────────────────────

_STALE_EXPORT_VARS = (
    "AIOS_DB_URL",
    "AIOS_API_PORT",
    "AIOS_INSTANCE_ID",
    "AIOS_WORKSPACE_ROOT",
)


def warn_stale_exports() -> None:
    """Warn if the caller's shell has aios runtime vars already exported.

    Bootstrap's own subprocess uses a hermetic env so its migration is
    never misdirected — but the user's subsequent ``aios api`` /
    ``aios worker`` will inherit the shell's env, which beats the
    worktree ``.env``. This is non-fatal; we just tell the user.
    """
    stale = [v for v in _STALE_EXPORT_VARS if v in os.environ]
    if not stale:
        return
    print_note(
        f"warning: {', '.join(stale)} is exported in your shell and will "
        f"override the worktree .env at runtime."
    )
    print_note("Start a fresh shell before running `aios api` / `aios worker`, or:")
    print_note(f"  unset {' '.join(_STALE_EXPORT_VARS)}")


# ── admin DB ops (async helpers) ───────────────────────────────────────────


async def _preflight_role_can_create_db(admin_url: str) -> None:
    """Connect as the admin role and verify it can CREATE DATABASE.

    A superuser bypasses the rolcreatedb flag, so either qualifier passes.
    """
    conn = await asyncpg.connect(admin_url)
    try:
        row = await conn.fetchrow(
            "SELECT rolsuper, rolcreatedb FROM pg_roles WHERE rolname = current_user"
        )
    finally:
        await conn.close()
    if row is None or not (row["rolsuper"] or row["rolcreatedb"]):
        raise RuntimeError(
            "Postgres role lacks CREATEDB/SUPERUSER. Grant with: ALTER USER <role> CREATEDB;"
        )


async def _create_database_if_missing(admin_url: str, db_name: str) -> bool:
    """Run ``CREATE DATABASE "<db_name>"``; treat DuplicateDatabaseError as no-op.

    Returns True if the database was created, False if it already existed.
    ``db_name`` MUST have been validated by :func:`validate_instance_id_for_db`
    prior to this call — it flows directly into an identifier position.
    """
    conn = await asyncpg.connect(admin_url)
    try:
        try:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            return True
        except asyncpg.exceptions.DuplicateDatabaseError:
            return False
    finally:
        await conn.close()


async def _drop_database(admin_url: str, db_name: str) -> None:
    """Issue ``DROP DATABASE "<db_name>"``. Raises on any Postgres error.

    If PG refuses because connections are open, the caller should catch
    the exception and list blocking PIDs via :func:`_list_db_backends`.
    ``db_name`` MUST be pre-validated.
    """
    conn = await asyncpg.connect(admin_url)
    try:
        await conn.execute(f'DROP DATABASE "{db_name}"')
    finally:
        await conn.close()


async def _list_db_backends(admin_url: str, db_name: str) -> list[tuple[int, str]]:
    """Return ``(pid, application_name)`` for each backend attached to ``db_name``."""
    conn = await asyncpg.connect(admin_url)
    try:
        rows = await conn.fetch(
            "SELECT pid, COALESCE(application_name, '') AS application_name "
            "FROM pg_stat_activity "
            "WHERE datname = $1 AND pid <> pg_backend_pid()",
            db_name,
        )
    finally:
        await conn.close()
    return [(int(r["pid"]), str(r["application_name"])) for r in rows]


async def _count_db_clients(db_url: str) -> int:
    """Count backends attached to the instance DB itself (dev status).

    Uses the instance DSN, not the admin DSN — ``pg_stat_activity``
    rows for a DB are visible from within that DB. Filters out the
    counting connection itself.
    """
    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.fetchval(
            "SELECT COUNT(*) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )
    finally:
        await conn.close()
    return int(result or 0)


def _resolve_admin_url(secrets: dict[str, str]) -> str | None:
    """Pick the admin DSN from process env, then parsed secrets — no default.

    Returns ``None`` when neither source supplies ``AIOS_POSTGRES_ADMIN_URL``.
    There is deliberately NO silent fallback (issue #824): the old default
    ``postgresql://aios:aios@localhost:5432/postgres`` could connect bootstrap/
    teardown to whatever owns :5432 on a multi-postgres host (e.g. a foreign
    production server), silently CREATE/DROP ``aios_dev_*`` databases there.
    Requiring an explicit DSN forces the operator to name the target server.

    Pure: process env takes precedence over the secrets.env value. Callers
    are responsible for turning ``None`` into a user-facing error.
    """
    return os.environ.get("AIOS_POSTGRES_ADMIN_URL") or secrets.get("AIOS_POSTGRES_ADMIN_URL")


def _print_missing_admin_url_error() -> None:
    """Explain how to supply the now-required admin DSN (issue #824)."""
    print_error("AIOS_POSTGRES_ADMIN_URL is not set.")
    print_note("Supply it one of two ways:")
    print_note("  export AIOS_POSTGRES_ADMIN_URL=postgresql://aios:aios@localhost:5433/postgres")
    print_note("  or add AIOS_POSTGRES_ADMIN_URL=... to ~/.aios/secrets.env")
    print_note(
        "This is required deliberately: there is no silent localhost:5432 default, "
        "so bootstrap/teardown never targets a foreign postgres on a multi-DB host."
    )


# ── docker helpers (shell out, no Settings import) ─────────────────────────


def _list_instance_containers(instance_id: str) -> list[str]:
    """Return container ids matching both managed+instance labels."""
    out = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            "label=aios.managed=true",
            "--filter",
            f"label=aios.instance_id={instance_id}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        print_error(f"docker ps failed: {out.stderr.strip()}")
        raise typer.Exit(1)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _force_remove_containers(container_ids: list[str]) -> None:
    if not container_ids:
        return
    rc = subprocess.call(
        ["docker", "rm", "--force", *container_ids],
        stdout=subprocess.DEVNULL,
    )
    if rc != 0:
        print_error(f"docker rm --force returned exit {rc}")
        raise typer.Exit(1)


# ── CLI sub-app ────────────────────────────────────────────────────────────

app = typer.Typer(
    name="dev",
    help="Manage a per-worktree aios dev instance.",
    no_args_is_help=True,
)


@app.command("bootstrap", help="Create a per-worktree aios dev instance.")
def bootstrap() -> None:
    repo_root = get_git_repo_root()
    instance_id = derive_instance_id(repo_root)
    validate_instance_id_for_db(instance_id)
    db_name = DB_PREFIX + instance_id

    warn_stale_exports()

    secrets_path = Path.home() / ".aios" / "secrets.env"
    secrets = parse_env_file(secrets_path)
    missing = [k for k in ("AIOS_BOOTSTRAP_TOKEN", "AIOS_VAULT_KEY") if k not in secrets]
    if missing:
        print_error(f"{secrets_path} is missing required keys: {', '.join(missing)}")
        print_note("Required contents (generate values yourself):")
        print_note("  AIOS_BOOTSTRAP_TOKEN=<openssl rand -hex 32>")
        print_note("  AIOS_VAULT_KEY=<openssl rand -base64 32>")
        print_note("Plus any provider keys (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, ...).")
        raise typer.Exit(1)

    admin_url = _resolve_admin_url(secrets)
    if admin_url is None:
        _print_missing_admin_url_error()
        raise typer.Exit(1)

    try:
        asyncio.run(_preflight_role_can_create_db(admin_url))
    except (asyncpg.exceptions.PostgresError, OSError, RuntimeError) as exc:
        print_error(f"Postgres admin preflight failed: {exc}")
        print_note(f"Admin DSN: {admin_url}")
        raise typer.Exit(1) from exc

    try:
        created = asyncio.run(_create_database_if_missing(admin_url, db_name))
    except asyncpg.exceptions.PostgresError as exc:
        print_error(f"CREATE DATABASE {db_name!r} failed: {exc}")
        raise typer.Exit(1) from exc
    print_note(f"database: {db_name} ({'created' if created else 'already exists'})")

    env_path = repo_root / ".env"
    existing_env = parse_env_file(env_path)
    if "AIOS_API_PORT" in existing_env:
        port = int(existing_env["AIOS_API_PORT"])
        if port_is_open(port):
            print_note(f"port {port} is currently in use; keeping it since it's persisted in .env")
    else:
        port = pick_free_port()

    instance_root = (Path.home() / ".aios" / "instances" / instance_id).resolve()
    workspace_root = instance_root / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)

    runtime_db_url = derive_runtime_db_url(admin_url, instance_id)

    write_env_atomic(
        env_path,
        {
            "AIOS_INSTANCE_ID": instance_id,
            "AIOS_DB_URL": runtime_db_url,
            "AIOS_API_PORT": str(port),
            "AIOS_WORKSPACE_ROOT": str(workspace_root),
        },
    )

    # Hermetic subprocess env: explicit overrides beat any stale shell exports.
    # Parsed secrets are passed through so the child doesn't need to re-read
    # ~/.aios/secrets.env (which pydantic-settings resolves at class-definition
    # time, not per-invocation).
    migrate_env = os.environ | {
        "AIOS_INSTANCE_ID": instance_id,
        "AIOS_DB_URL": runtime_db_url,
        "AIOS_API_PORT": str(port),
        "AIOS_WORKSPACE_ROOT": str(workspace_root),
        "AIOS_BOOTSTRAP_TOKEN": secrets["AIOS_BOOTSTRAP_TOKEN"],
        "AIOS_VAULT_KEY": secrets["AIOS_VAULT_KEY"],
    }
    rc = subprocess.call(
        ["uv", "run", "aios", "migrate"],
        cwd=str(repo_root),
        env=migrate_env,
    )
    if rc != 0:
        print_error(f"aios migrate failed (exit {rc})")
        raise typer.Exit(rc)

    print_note("")
    print_note(f"Bootstrapped instance={instance_id} db={db_name} port={port}")
    print_note("")
    print_note("IMPORTANT: start a fresh shell, or explicitly unset any stale aios vars:")
    print_note(f"  unset {' '.join(_STALE_EXPORT_VARS)}")
    print_note("")
    print_note("Then activate this worktree:")
    print_note("  set -a && source .env && set +a")
    print_note("")
    print_note("Run:")
    print_note("  uv run aios api       # terminal 1")
    print_note("  uv run aios worker    # terminal 2")
    print_note("")
    print_note("Mint the root account's first API key (one-shot per DB), then use it:")
    print_note("  export AIOS_API_KEY=$(uv run aios -f json accounts bootstrap \\")
    print_note(
        "    | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"plaintext_key\"])')"
    )
    print_note("  uv run aios agents list")


@app.command(
    "teardown",
    help="Drop this worktree's aios dev database, containers, and workspace.",
)
def teardown(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Bypass the API-port liveness check. Does NOT terminate DB backends.",
        ),
    ] = False,
) -> None:
    repo_root = get_git_repo_root()
    env = load_instance_env(repo_root)
    instance_id = env["AIOS_INSTANCE_ID"]
    validate_instance_id_for_db(instance_id)
    db_name = DB_PREFIX + instance_id
    port = int(env["AIOS_API_PORT"])
    workspace_root = Path(env["AIOS_WORKSPACE_ROOT"])

    if port_is_open(port) and not force:
        print_error(
            f"aios api appears to be running on :{port}. "
            f"Stop it first, or pass --force to bypass this check."
        )
        raise typer.Exit(1)

    container_ids = _list_instance_containers(instance_id)

    if not yes:
        print_note(f"About to destroy aios dev instance {instance_id}:")
        print_note(f"  - DROP DATABASE {db_name}")
        print_note(f"  - remove {len(container_ids)} container(s)")
        print_note(f"  - rm -rf {workspace_root}")
        confirm = typer.prompt("Proceed? (yes/no)", default="no")
        if confirm.strip().lower() not in ("yes", "y"):
            print_note("aborted")
            raise typer.Exit(1)

    _force_remove_containers(container_ids)

    secrets = parse_env_file(Path.home() / ".aios" / "secrets.env")
    admin_url = _resolve_admin_url(secrets)
    if admin_url is None:
        _print_missing_admin_url_error()
        raise typer.Exit(1)
    try:
        asyncio.run(_drop_database(admin_url, db_name))
    except asyncpg.exceptions.PostgresError as exc:
        print_error(f"DROP DATABASE {db_name!r} failed: {exc}")
        try:
            backends = asyncio.run(_list_db_backends(admin_url, db_name))
        except asyncpg.exceptions.PostgresError:
            backends = []
        if backends:
            print_note("Blocking backends:")
            for pid, app_name in backends:
                print_note(f"  pid={pid} application_name={app_name!r}")
            print_note("Stop the worker/api attached to this instance, then re-run teardown.")
        raise typer.Exit(1) from exc

    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    print_note(f"teardown complete: instance={instance_id}")


@app.command(
    "shutdown",
    help="Stop this worktree's aios api + worker processes and pause its sandbox containers.",
)
def shutdown(
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet", help="Suppress status output (used by Claude Code SessionEnd hook)."
        ),
    ] = False,
) -> None:
    """Stop processes and pause containers, leaving DB and workspace intact.

    Idempotent — safe to run repeatedly; nothing to stop is a clean exit.

    Used by the ``SessionEnd`` hook in ``.claude/settings.json`` so a
    closed Claude Code session doesn't leave orphan api/worker processes
    or running sandbox containers behind. See #311.
    """
    repo_root = get_git_repo_root()
    # Pre-check the .env directly rather than calling load_instance_env
    # — the latter raises typer.Exit on missing keys, which is the right
    # behavior for teardown/status but the wrong one for an idempotent
    # shutdown that should silently no-op when no instance exists.
    env = parse_env_file(repo_root / ".env")
    instance_id = env.get("AIOS_INSTANCE_ID")
    if not instance_id:
        if not quiet:
            print_note("no aios dev instance for this worktree (idempotent no-op)")
        return
    try:
        validate_instance_id_for_db(instance_id)
    except ValueError as exc:
        # ``shutdown`` runs from the SessionEnd hook; a malformed env value
        # shouldn't make the hook crash on its way out of a Claude Code
        # session. Print a quiet note and exit clean.
        if not quiet:
            print_note(f"AIOS_INSTANCE_ID looks malformed; skipping shutdown ({exc})")
        return

    # pkill matches against the full command line via `-f`. Match the
    # worktree path so we don't accidentally kill processes belonging to
    # a sibling worktree's instance.
    worktree_path = str(repo_root)
    api_killed = _pkill_pattern(f"{worktree_path}.*-m aios api")
    worker_killed = _pkill_pattern(f"{worktree_path}.*-m aios worker")

    container_ids = _list_instance_containers(instance_id)
    if container_ids:
        # Stop, don't remove — `teardown` is the destructive path. A
        # subsequent `aios api` / `aios worker` will respawn fresh
        # sandboxes from the recorded labels.
        subprocess.call(
            ["docker", "stop", *container_ids],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if not quiet:
        print_note(
            f"shutdown: instance={instance_id} "
            f"api={'stopped' if api_killed else 'not-running'} "
            f"worker={'stopped' if worker_killed else 'not-running'} "
            f"containers_stopped={len(container_ids)}"
        )


def _pkill_pattern(pattern: str) -> bool:
    """SIGTERM every process whose full cmdline matches ``pattern``.

    Returns ``True`` if pkill found at least one match (exit code 0),
    ``False`` if no match (exit code 1). Other exit codes are treated
    as "no match" — pkill on macOS / Linux returns 1 for "no processes
    found" which is the dominant case for idempotent shutdown.
    """
    rc = subprocess.call(
        ["pkill", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


@app.command("status", help="Show this worktree's aios dev instance status.")
def status() -> None:
    repo_root = get_git_repo_root()
    # Read the worktree .env directly rather than via load_instance_env, which
    # hard-exits on missing keys — we must report the unbootstrapped case, not
    # die on it. Resolve the runtime DB the way pydantic will (exported value
    # beats .env) so a stale exported AIOS_DB_URL is reflected — that divergence
    # is the exact footgun #349 is about.
    env = parse_env_file(repo_root / ".env")
    resolved_db_url = runtime_db_url(repo_root)
    in_worktree = is_linked_worktree()
    mode = classify_instance_mode(env, resolved_db_url, in_worktree)

    print(f"mode:            {mode}")

    instance_id = env.get("AIOS_INSTANCE_ID")
    print(f"instance_id:     {instance_id or '-'}")

    if instance_id:
        print(f"database:        {DB_PREFIX + instance_id}")
    else:
        print("database:        -")

    port_str = env.get("AIOS_API_PORT")
    print(f"api_port:        {port_str or '-'}")

    workspace_root = env.get("AIOS_WORKSPACE_ROOT")
    print(f"workspace_root:  {workspace_root or '-'}")

    if instance_id:
        container_ids = _list_instance_containers(instance_id)
        print(f"containers:      {len(container_ids)}")
    else:
        print("containers:      -")

    if port_str:
        api_up = port_is_open(int(port_str))
        print(f"api_port_open:   {api_up}")
    else:
        print("api_port_open:   -")

    db_url = env.get("AIOS_DB_URL")
    if db_url:
        try:
            clients = asyncio.run(_count_db_clients(db_url))
            print(f"db_attached:     {clients}  (best-effort; any long-lived connection counts)")
        except (asyncpg.exceptions.PostgresError, OSError) as exc:
            print(f"db_attached:     unknown ({exc})")
    else:
        print("db_attached:     -")
