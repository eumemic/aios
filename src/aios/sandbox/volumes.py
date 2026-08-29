"""Per-session workspace directory conventions.

Each session gets a stable host-side directory at
``settings.workspace_root / session_id``, which is bind-mounted into the
session's container at ``/workspace``. The directory is created on the
first tool call that provisions a container for the session — chat-only
sessions never create one — and persists across container lifetimes
(container death, session re-wake, worker restart).

The directory is NOT deleted when the container goes away. A session can
be resumed tomorrow and still find its files. Cleanup of stale workspace
dirs is a Phase 6 polish item.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import stat
from pathlib import Path

from aios.config import get_settings
from aios.errors import ForbiddenError
from aios.logging import get_logger

log = get_logger("aios.sandbox.volumes")

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\-]")
_MAX_FILENAME_LEN = 200
_FILENAME_FALLBACK = "unnamed"


def safe_filename(name: str | None) -> str:
    """Sanitize ``name`` for use as a path leaf.

    Strips directory separators (defeats ``../`` traversal), maps
    unsupported characters to ``_``, falls back to ``"unnamed"`` for
    None / empty / all-dot inputs, and caps length so a pathological
    filename combined with a per-file id prefix can't exhaust the
    host FS's per-component limit.

    Unicode-aware: Python's ``\\w`` matches the full Unicode word class
    by default, so non-ASCII letters are preserved (e.g.
    ``café.jpg``, ``图片.png``). Only structurally unsafe punctuation
    and whitespace get rewritten.
    """
    if not name:
        return _FILENAME_FALLBACK
    base = os.path.basename(name)
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", base)
    if not cleaned or cleaned.replace(".", "") == "":
        return _FILENAME_FALLBACK
    return cleaned[:_MAX_FILENAME_LEN]


def ensure_owned_dir(path: Path) -> Path:
    """mkdir(parents=True, exist_ok=True) ``path``, then — iff running as root —
    chown every component newly created by THIS call (intermediates included) to
    the configured workspaces owner uid:gid. Non-root callers (api at uid 1000,
    dev single-uid) get exactly today's plain-mkdir behavior.

    Fixes #959: the worker (root) and api (uid 1000) both write under
    ``workspace_root``. A shared dir the worker creates first is ``root:root``
    and the api (no CAP_CHOWN) can't write into it. Chowning the components
    this call creates keeps the shared tree writable by the api owner.
    """
    settings = get_settings()
    root = settings.workspace_root.resolve()
    target = path.resolve()
    newly: list[Path] = []
    cur = target
    while True:
        if cur.exists():
            break
        newly.append(cur)
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    path.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        uid, gid = settings.workspaces_owner_uid, settings.workspaces_owner_gid
        for component in newly:
            # Only chown within the workspace tree (workspace_root itself and its
            # descendants); never touch out-of-tree ancestors. If a caller passed
            # a path outside the tree (the mkdir honors any path — caller's
            # contract), the component-walk collected out-of-tree ancestors; do
            # NOT chown those.
            if not component.is_relative_to(root):
                continue
            # A racing provision may have created+chowned this component first
            # (benign), but the failure must stay observable per CLAUDE.md's
            # no-silent-error stance — log and continue rather than crash.
            try:
                # lchown (not chown): closes the mkdir→chown symlink-swap race —
                # a container with workspace write access could replace a
                # freshly-created component with a symlink before we chown,
                # redirecting os.chown to an out-of-tree target. Matches the
                # repair path's lchown.
                os.lchown(component, uid, gid)
            except OSError as e:
                log.warning("workspace.chown_failed", path=str(component), error=str(e))
    return path


# DEPRECATED post-#409 — do not use in new code; see issue #630.
def workspace_dir_for(session_id: str) -> Path:
    """Return the absolute host directory for ``session_id``'s workspace.

    The returned path is always absolute — Docker bind mounts reject
    relative paths. If ``workspace_root`` was configured as a relative
    path (e.g. ``./workspaces`` in a dev ``.env``), it is resolved
    against the current working directory at call time.

    Pure — does not touch the filesystem.
    """
    return (get_settings().workspace_root / session_id).resolve()


def ensure_workspace_path(raw_path: str) -> Path:
    """Resolve ``raw_path`` to an absolute ``Path``, creating it if needed."""
    path = Path(raw_path).resolve()
    return ensure_owned_dir(path)


def validate_workspace_path(
    raw_path: str, account_id: str, *, session_id: str | None = None
) -> None:
    """Refuse ``raw_path`` if it resolves outside the account's
    workspace subdirectory.

    ``raw_path`` MUST be absolute.  Relative inputs are rejected
    before any ``Path.resolve()`` runs: ``resolve()`` would
    interpret them against the current process's working directory,
    which differs between the API and worker (and between worker
    restarts), producing diverging targets across boundaries.  See
    #626 — legacy session rows persisted relative
    ``workspaces/<account>/<session>`` strings (back when
    ``AIOS_WORKSPACE_ROOT`` itself was permitted to be relative);
    every cold-start re-validation surfaced ``ForbiddenError``
    blamed on whatever path the model had just tried to use.
    Failing fast on the relative-input branch produces an
    unambiguous error identifying the stored
    ``workspace_volume_path`` as the culprit.

    Without this check an authenticated client could POST
    ``/v1/sessions`` with e.g. ``workspace_path="/etc"`` and the
    sandbox would bind-mount the host's ``/etc`` read-write at
    ``/workspace`` — arbitrary host filesystem read/write via any
    bash / write / edit tool call.  A path under another account's
    subdir (``workspace_root/{other_account_id}/...``) would defeat
    the per-account-subdir isolation ``insert_session`` enforces by
    default.

    ``Path.resolve()`` collapses ``..`` traversal and dereferences
    symlinks on the supplied path before the ``is_relative_to``
    check, so create-time inputs that already point outside the jail
    are rejected — including ``/etc``, ``..``-traversal back up to
    ``workspace_root/{other_account_id}``, and symlinks under the
    account's subdir that point outward at validate time.

    ``session_id`` opens a tiny backward-compat carve-out for the
    pre-#409 default ``<workspace_root>/<session_id>`` (no per-tenant
    subdir).  Sessions created before PR #409 have ``workspace_volume_path``
    rows in exactly that shape; without the carve-out the bind-mount
    boundary re-check (added by PR #590) rejects them on every
    cold-start, leaving the model staring at a ``ForbiddenError`` on
    every tool call.  The carve-out is keyed on the session_id the
    caller is currently provisioning — a path matching the legacy shape
    but naming a *different* session_id is still rejected, so the
    cross-tenant defense holds.  The carve-out also requires the
    resolved path to remain under ``workspace_root``: a symlink at
    ``<workspace_root>/<session_id>`` pointing at ``/etc`` (or any
    other path outside the jail) is rejected, preserving the
    host-FS-escape defense the strict branch provides.  The create-time
    call sites leave ``session_id`` unset so user-supplied paths remain
    strictly jailed to the account subdir.

    Limitations: this is the create-time + bind-mount-time check on
    the workspace_path argument.  Symlinks WRITTEN inside the
    mounted ``/workspace`` after the bind-mount is live still resolve
    on the host filesystem at access time (Docker bind-mount semantic),
    so a tool ``ln -s /etc /workspace/sneaky`` followed by
    ``cat /workspace/sneaky/passwd`` still reads host ``/etc/passwd``.
    Fencing that surface requires kernel-level mount options
    (``nosymfollow``) or container-level MAC; out of scope for this
    fix.

    Raises ``ForbiddenError`` (403, not 422): semantically this is an
    attempted privilege escalation per the project's "fail hard" stance,
    and surfacing it under the auth-tier error family makes it visible
    in audit logs as such.
    """
    if not Path(raw_path).is_absolute():
        raise ForbiddenError(
            "workspace_volume_path must be absolute (starts with '/'); got "
            f"non-absolute value {raw_path!r}. This usually indicates a "
            "stale pre-#409 session row that needs the absolute-legacy "
            "backfill migration (see aios#626).",
            detail={"workspace_path": raw_path, "session_id": session_id},
        )
    path = Path(raw_path).resolve()
    workspace_root = get_settings().workspace_root.resolve()
    account_root = (workspace_root / account_id).resolve()
    if path.is_relative_to(account_root):
        return
    if session_id is not None:
        legacy_path = (workspace_root / session_id).resolve()
        if path == legacy_path and legacy_path.is_relative_to(workspace_root):
            return
    raise ForbiddenError(
        "workspace_path must resolve to within the account's workspace subdirectory",
        detail={"workspace_path": raw_path},
    )


_RUNS_ROOT = "_runs"


def run_workspace_dir(account_id: str, run_id: str) -> Path:
    """Per-run host workspace directory backing ``/workspace`` in a run sandbox.

    Run scratch is account-scoped at
    ``<workspace_root>/<account_id>/_runs/<run_id>``. Pure — does not touch the
    filesystem; use :func:`ensure_run_workspace_dir` to create it.
    """
    return (get_settings().workspace_root / account_id / _RUNS_ROOT / run_id).resolve()


def ensure_run_workspace_dir(account_id: str, run_id: str) -> Path:
    """Return the per-run workspace directory, creating it and its parents."""
    return ensure_owned_dir(run_workspace_dir(account_id, run_id))


_BROWSER_ROOT = "_browser"


def _plane_subdirs_from_protocol() -> tuple[str, ...]:
    """Derive the plane subdirectories from the wire contract.

    The driver's container-side paths (``browser_protocol`` — the single
    authority for the plane layout, COPY'd into the browser image) determine
    which host-side subdirs must exist: ``profile`` (Chromium user data —
    logins; non-reconstructible, never auto-reaped, deleted only by explicit
    clear-state), ``shots``, ``frames``, ``downloads``, and ``input`` (the
    takeover input spool). Deriving rather than restating means a renamed
    subdir cannot ship a driver writing where the host never created a dir.
    """
    from pathlib import PurePosixPath

    from aios.sandbox import browser_protocol as proto

    paths = (
        proto.PROFILE_DIR,
        proto.SHOTS_DIR,
        proto.FRAMES_DIR,
        proto.DOWNLOADS_DIR,
        proto.INPUT_SPOOL,
    )
    return tuple(PurePosixPath(p).relative_to("/workspace").parts[0] for p in paths)


BROWSER_PLANE_SUBDIRS = _plane_subdirs_from_protocol()


def browser_plane_root() -> Path:
    """``<workspace_root>/_browser`` — the parent of every account plane."""
    return (get_settings().workspace_root / _BROWSER_ROOT).resolve()


def browser_plane_dir(account_id: str) -> Path:
    """Per-account browser-plane host directory: ``<workspace_root>/_browser/<account_id>``.

    Deliberately TOP-LEVEL — a sibling of the account workspace dirs, not a
    child: :func:`validate_workspace_path` jails user-supplied session
    workspaces to ``<workspace_root>/<account_id>/…``, so a plane inside the
    account subdir could be bind-mounted read-write into an agent sandbox via
    a session's ``workspace_path`` (cookie theft). Outside every account jail,
    no session workspace can resolve into it by construction (jarbot#106
    §6.2). Pure — use :func:`ensure_browser_plane_dir` to create it.
    """
    return browser_plane_root() / account_id


def ensure_browser_plane_dir(account_id: str) -> Path:
    """Return the per-account browser plane dir, creating it and its subdirs."""
    plane = ensure_owned_dir(browser_plane_dir(account_id))
    for sub in BROWSER_PLANE_SUBDIRS:
        ensure_owned_dir(plane / sub)
    return plane


# Per-read byte ceiling for plane files. The plane is written by a
# (potentially compromised) container, so an unbounded read lets it plant one
# oversized file and have the 5 Hz frame poll re-read it into the shared API
# process — memory/CPU exhaustion. 64 MiB is far above any legitimate frame
# (JPEG q70, <1 MiB) or screenshot yet caps the amplification. A fixed
# constant, never operator-tunable to an unsafe value.
_PLANE_READ_MAX_BYTES = 64 * 1024 * 1024


def read_plane_file(plane: Path, rel: str) -> bytes | None:
    """Read ``plane/rel`` following NO symlink at ANY path component.

    The plane is a bind mount a (potentially compromised) browser container
    writes, so any resolve-then-check-then-read sequence is a TOCTOU hole:
    the container can swap a checked component for a symlink into another
    account's plane between the check and the read (the 5 Hz frame poll
    hands it unlimited attempts — jarbot#106 Phase 2 red-team F1). Here
    every component is opened relative to its parent's directory fd with
    ``O_NOFOLLOW``, so a symlink anywhere fails ``ELOOP`` instead of being
    followed, and containment inside ``plane`` holds by construction (no
    absolute refs, no ``..``, no empty or NUL-bearing components). The leaf
    open adds ``O_NONBLOCK`` so a FIFO planted at the path cannot wedge the
    reader (regular-file reads ignore the flag), ``fstat`` rejects anything
    that is not a regular file, and the read is bounded by
    ``_PLANE_READ_MAX_BYTES`` (checked against the running total, so a file
    grown after the open cannot exceed it either).

    Returns ``None`` for any malformed / missing / symlinked / non-regular /
    oversized path — never raises (the screenshot sink treats a raised
    exception as the CALLING session's sandbox being unhealthy and evicts it,
    which a hostile ref must not be able to trigger). Malformed refs and
    symlinks are logged — they only occur under a hostile or corrupted
    container and operators want the signal; plain ``ENOENT`` stays quiet
    (normal during frame rotation).
    """
    parts = rel.split("/")
    # An absolute ref splits to a leading "" component, so the empty-component
    # check subsumes the leading-slash case. NUL bytes would make os.open raise
    # ValueError (not OSError) — refuse them here so the contract stays
    # never-raises.
    if any(part in ("", ".", "..") or "\x00" in part for part in parts):
        log.warning("plane.read_ref_malformed", plane=str(plane), rel=rel)
        return None
    try:
        fd = os.open(plane, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except OSError as err:
        if err.errno != errno.ENOENT:
            log.warning("plane.read_refused", plane=str(plane), rel=rel, error=str(err))
        return None
    finally:
        os.close(fd)
    try:
        if not stat.S_ISREG(os.fstat(leaf).st_mode):
            log.warning("plane.read_not_regular_file", plane=str(plane), rel=rel)
            return None
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(leaf, 1 << 20):
            chunks.append(chunk)
            total += len(chunk)
            if total > _PLANE_READ_MAX_BYTES:
                log.warning("plane.read_too_large", plane=str(plane), rel=rel, bytes=total)
                return None
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(leaf)


_MEMORY_STORES_ROOT = "_memory_stores"


def memory_stores_root() -> Path:
    """Return ``<workspace_root>/_memory_stores`` — the parent of all
    shared memory-store host directories.

    Per-store host dirs (one per ``memory_store.id``) live as siblings in
    here and are bind-mounted into every attached session's container at
    ``/mnt/memory/<store_name>/``. Sharing the source dir across attached
    sessions is what makes cross-session reads live: a tool write from
    session A appears in session B's mount immediately.
    """
    return (get_settings().workspace_root / _MEMORY_STORES_ROOT).resolve()


def memory_store_host_dir(store_id: str) -> Path:
    """Return the shared host-side directory backing memory store ``store_id``.

    Pure — does not create the directory. Materialization is handled by
    :mod:`aios.sandbox.memory_mounts`, which acquires the matching lock
    file (see :func:`memory_store_lock_path`) before populating from DB.
    """
    return memory_stores_root() / store_id


def memory_store_lock_path(store_id: str) -> Path:
    """Return the file-lock path used to serialize first-attach materialization
    of ``store_id``.

    Two sessions provisioning concurrently for the same store both call
    :func:`materialize_store_to_host`; the lock ensures only one of them
    writes the initial DB snapshot to the host dir. The loser observes
    the ``.materialized`` marker and skips."""
    return memory_stores_root() / f"{store_id}.lock"


_GITHUB_REPOS_ROOT = "_github_repos"


def github_repos_cache_root() -> Path:
    """Return ``<workspace_root>/_github_repos`` — the parent of all
    cache-bare-clone host directories.

    Each cache dir is keyed by ``sha256(repo_url)`` so two sessions that
    reference the same upstream repo share the object database, regardless
    of ``mount_path``. Per-session working trees ``--reference`` this cache
    via :func:`session_repo_working_tree_dir`.
    """
    return (get_settings().workspace_root / _GITHUB_REPOS_ROOT).resolve()


def github_repo_cache_dir(url_hash: str) -> Path:
    """Bare-clone host dir for a given repo-url hash. Pure — see
    :mod:`aios.sandbox.github_clone` for materialization."""
    return github_repos_cache_root() / url_hash


def github_repo_cache_lock_path(url_hash: str) -> Path:
    """Per-cache file lock path. Two sessions racing on first-clone of the
    same url need to serialize so we don't run two ``git clone --bare``
    side by side and corrupt the cache dir."""
    return github_repos_cache_root() / f"{url_hash}.lock"


_SESSION_REPOS_ROOT = "_session_repos"


def session_repos_root(session_id: str) -> Path:
    """Per-session host dir for github_repository working trees.

    Rooted at ``<workspace_root>/_session_repos/<session_id>`` so the
    location is independent of any user-supplied ``workspace_path``
    override on the session — those overrides are user-managed and we
    don't want plaintext-token ``.git/config`` files leaking into them.
    """
    return (get_settings().workspace_root / _SESSION_REPOS_ROOT / session_id).resolve()


def session_repo_working_tree_dir(session_id: str, repo_id: str) -> Path:
    """Per-session working tree for a single ``github_repository``
    attachment. Bind-mounted into the container at the user-supplied
    ``mount_path``.
    """
    return session_repos_root(session_id) / repo_id


_SESSION_TMP_ROOT = "_tmp"


def session_tmp_root() -> Path:
    """Return ``<workspace_root>/_tmp`` — the parent of all per-session
    ephemeral-scratch directories."""
    return (get_settings().workspace_root / _SESSION_TMP_ROOT).resolve()


def session_tmp_dir(session_id: str) -> Path:
    """Per-session host directory bind-mounted into the container at ``/tmp``.

    ``/tmp`` used to be a plain overlay directory inside the sandbox, which
    made it *durable state*: ``docker export``/``docker commit`` faithfully
    copied every byte of it into the session's snapshot image. Long-lived
    sessions accumulate per-task scratch there — repo clones, virtualenvs,
    pytest temp trees, build caches — none of which is worth preserving and
    all of which was paid for twice (once as the compressed content blob,
    once as the unpacked snapshot) on every idle reap. One production
    session reached 18 GiB of which 16.4 GiB was ``/tmp`` (eumemic/aios#2280).

    Binding ``/tmp`` to a host directory fixes that by construction: Docker
    excludes bind-mount contents from BOTH snapshot verbs, so ephemeral
    scratch can never enter an image again — no filter, no allowlist, no
    per-call opt-in to forget.

    The scratch lives on the workspace volume beside the other reaped trees,
    so it is separately provisioned, individually attributable
    (``du -sh <workspace_root>/_tmp/<session_id>``), and reclaimed by
    :mod:`aios.harness.host_dir_reaper` once the session is no longer live.
    It survives container recycles, which is strictly more continuity than
    the pre-mount behaviour gave a cold-started sandbox.

    Rooted at ``<workspace_root>/_tmp/<session_id>`` rather than under the
    session's own ``workspace_path`` for the same reason as
    :func:`session_repos_root`: that path is user-supplied, is not unique
    per session, and must not become a place where scratch lands inside
    data the user manages.

    Pure — does not create the directory. Use :func:`ensure_session_tmp_dir`.
    """
    return session_tmp_root() / session_id


def ensure_session_tmp_dir(session_id: str) -> Path:
    """Return the per-session ``/tmp`` backing directory, creating it if needed.

    Called from the spec builder at every container start so the bind-mount
    source always exists before Docker tries to mount it. Docker would
    otherwise create a missing source as ``root:root``, which the api (uid
    1000) could not write into — the #959 failure mode ``ensure_owned_dir``
    exists to prevent.

    The directory is chmod 1777 (world-writable + sticky) to match the
    ``/tmp`` semantics every tool in the sandbox assumes: sandbox processes
    do not all run as the owning uid, and a non-sticky world-writable dir
    would let one of them delete another's files.
    """
    path = ensure_owned_dir(session_tmp_dir(session_id))
    try:
        path.chmod(0o1777)
    except OSError:
        # Best-effort: a pre-existing dir owned by another uid (worker root
        # vs api 1000) can refuse chmod. The mount still works; only the
        # permission hardening is skipped. Never fail a provision over it.
        log.warning("could not chmod session tmp dir to 1777", path=str(path))
    return path


_ATTACHMENTS_ROOT = "_attachments"


def attachments_root() -> Path:
    """Return ``<workspace_root>/_attachments`` — the parent of all
    per-session inbound attachment directories.

    Each session subdir is bind-mounted read-only into its container at
    ``/mnt/attachments`` (see :mod:`aios.sandbox.provisioner`). Inbound
    binary blobs (Signal photos, Telegram voice notes, etc.) are staged
    here by :mod:`aios.services.attachment_staging` from
    :func:`aios.services.inbound.handle_inbound` before the inbound
    event is appended; the model sees them at stable in-sandbox paths
    of the form ``/mnt/attachments/<connector>/<event-ulid>-<filename>``.
    """
    return (get_settings().workspace_root / _ATTACHMENTS_ROOT).resolve()


def session_attachments_dir(session_id: str) -> Path:
    """Per-session host directory backing ``/mnt/attachments``.

    Pure — does not create the directory. Use
    :func:`ensure_session_attachments_dir` to create.
    """
    return attachments_root() / session_id


def ensure_session_attachments_dir(session_id: str) -> Path:
    """Return the per-session attachments directory, creating it if needed.

    Called eagerly from the provisioner at every container start so the
    bind-mount source always exists before Docker tries to mount it,
    even for sessions that have never received an attachment.
    """
    path = session_attachments_dir(session_id)
    return ensure_owned_dir(path)


_UPLOADS_ROOT = "_uploads"


def uploads_root() -> Path:
    """Return ``<workspace_root>/_uploads`` — the parent of all per-session
    upload directories.

    Each session subdir is bind-mounted read-only into its container at
    ``/mnt/uploads`` (see :mod:`aios.sandbox.spec`). Bytes uploaded via
    ``POST /v1/sessions/<id>/files`` land here under
    ``<workspace_root>/_uploads/<session_id>/<file_id>/<filename>``; the
    model sees them at ``/mnt/uploads/<file_id>/<filename>``.
    """
    return (get_settings().workspace_root / _UPLOADS_ROOT).resolve()


def session_uploads_dir(session_id: str) -> Path:
    """Per-session host directory backing ``/mnt/uploads``.

    Pure — does not create the directory. Use
    :func:`ensure_session_uploads_dir` to create.
    """
    return uploads_root() / session_id


def ensure_session_uploads_dir(session_id: str) -> Path:
    """Return the per-session uploads directory, creating it if needed.

    Called eagerly from the provisioner at every container start so the
    bind-mount source always exists before Docker tries to mount it,
    even for sessions that have never received an upload.
    """
    path = session_uploads_dir(session_id)
    return ensure_owned_dir(path)


def resolve_to_host_path(
    session_id: str,
    sandbox_path: str,
    *,
    workspace_path: Path | None = None,
) -> Path | None:
    """Map an in-sandbox path to its host-side equivalent for known bind mounts.

    Returns ``None`` when:

    * ``sandbox_path`` doesn't resolve into ``/workspace`` or
      ``/mnt/attachments`` (e.g. ``/etc/hostname``, ``/mnt/memory/...``,
      ``/tmp/...``), or
    * the resolved candidate escapes the bind-mount root after ``..``
      normalization or symlink dereferencing.

    ``workspace_path`` is the actual host-side bind-mount source for
    ``/workspace`` (recorded on the session row as
    ``workspace_volume_path``).  Post-PR-#409 it lives at
    ``<workspace_root>/<account_id>/<session_id>``; pre-#409 sessions
    had it at ``<workspace_root>/<session_id>``.  Either way, the
    caller is the authority — when ``workspace_path`` is ``None``,
    ``/workspace*`` paths fail closed (return ``None``) rather than
    silently falling back to a synthetic path that might no longer be
    the bind-mount source.  The ``/mnt/attachments`` and
    ``/mnt/uploads`` branches ignore ``workspace_path`` — their
    locations are derived from ``session_id`` alone (see
    :func:`session_attachments_dir` / :func:`session_uploads_dir`).

    The containment check defends model-controlled callers (notably the
    image branch of :mod:`aios.tools.read`): without it, a path like
    ``/workspace/../../etc/hostname`` or a symlink at
    ``/workspace/sneaky.jpg`` pointing outside the bind mount would
    let the model read arbitrary host files the worker can access.
    """
    base, suffix = _bind_mount_base(session_id, sandbox_path, workspace_path=workspace_path)
    if base is None:
        return None
    candidate = base if suffix is None else base / suffix
    try:
        resolved = candidate.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
    except OSError:
        return None
    if resolved != resolved_base and not resolved.is_relative_to(resolved_base):
        return None
    return resolved


def _bind_mount_base(
    session_id: str,
    sandbox_path: str,
    *,
    workspace_path: Path | None = None,
) -> tuple[Path | None, str | None]:
    """Return the host base dir + remainder for ``sandbox_path``.

    ``(None, None)`` when the path doesn't fall under a known bind
    mount, or when ``sandbox_path`` is under ``/workspace`` and
    ``workspace_path`` is ``None`` (fail-closed).  ``(base, None)``
    when the path is exactly the root.
    """
    if sandbox_path == "/workspace":
        if workspace_path is None:
            return None, None
        return workspace_path.resolve(), None
    if sandbox_path.startswith("/workspace/"):
        if workspace_path is None:
            return None, None
        return workspace_path.resolve(), sandbox_path[len("/workspace/") :]
    if sandbox_path == "/mnt/attachments":
        return session_attachments_dir(session_id), None
    if sandbox_path.startswith("/mnt/attachments/"):
        return session_attachments_dir(session_id), sandbox_path[len("/mnt/attachments/") :]
    if sandbox_path == "/mnt/uploads":
        return session_uploads_dir(session_id), None
    if sandbox_path.startswith("/mnt/uploads/"):
        return session_uploads_dir(session_id), sandbox_path[len("/mnt/uploads/") :]
    return None, None


def _purge_target_if_owned(
    candidate: Path,
    *,
    session_id: str,
    owned_bases: tuple[Path, ...],
    root: Path,
) -> Path | None:
    """Return ``candidate``'s resolved path if it is provably a directory
    owned by ``session_id`` alone, else ``None`` — meaning *do not delete it*.

    Two independent conditions, both on the RESOLVED (symlink-dereferenced,
    ``..``-collapsed) real path — never on the raw string:

    1. **In jail.** The target must live strictly under ``workspace_root``.
       ``workspace_root`` itself is not owned.
    2. **Owned by this session.** The target must be one of the session's own
       canonical directories (or something strictly inside one).  Containment
       is checked against a base that is itself derived from ``session_id``
       and re-checked for jail residency, so a symlinked canonical dir can't
       launder an out-of-jail target through condition 2.

    Condition 2 is the one that matters and the one that was missing: a path
    that is in-jail but too HIGH in the tree — most importantly the account
    root ``<workspace_root>/<account_id>``, which ``validate_workspace_path``
    accepts at create time because ``is_relative_to`` is REFLEXIVE — is
    shared with every other live session of that tenant.  ``rmtree``-ing it
    on one session's delete destroys every sibling session's workspace.  An
    ancestor is not ownership; only a proven per-session location is.

    **Skip, don't raise.**  Refusing the ``rmtree`` is the safety property;
    aborting the *delete* is not.  ``delete_session`` calls this AFTER the
    session row is already committed as deleted, so raising would leave the
    caller with a deleted row and a 403 — and would break a legitimate,
    reachable shape: a workflow ``agent()`` child spawned with the default
    ``workspace='shared'`` stores the RUN's shared workspace
    (``<root>/_runs/<run_id>``, or the launcher session's dir)
    as its own ``workspace_volume_path``.  That directory is genuinely not
    the child's to delete — it belongs to the run and is shared with the
    parent and every sibling child — so skipping it is exactly right, while
    raising turned a correct refusal into a failed deletion. Shared ``_runs``
    directories have their own scratch lifecycle. A skipped custom path on a
    deleted session, however, has no row left for the archived-workspace reaper
    to discover: that is a deliberate storage leak in preference to
    irrecoverable cross-session data loss.
    """
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        log.warning(
            "refusing to purge session directory outside workspace_root",
            path=str(candidate),
            session_id=session_id,
        )
        return None
    for base in owned_bases:
        base_resolved = base.resolve()
        # The base must itself be in-jail: otherwise a symlink AT the
        # canonical location would make an out-of-jail target "contained".
        if base_resolved == root or not base_resolved.is_relative_to(root):
            continue
        if resolved == base_resolved or resolved.is_relative_to(base_resolved):
            return resolved
    log.warning(
        "refusing to purge a directory that is not exclusively owned by this session",
        path=str(candidate),
        session_id=session_id,
    )
    return None


def purge_session_directories(
    session_id: str,
    workspace_path: Path,
    *,
    account_id: str,
    live_workspace_paths: tuple[str, ...] = (),
) -> None:
    """Remove every host directory exclusively owned by ``session_id``.

    Every candidate is resolved and proven to be *the session's own*
    directory before anything is deleted.  This is intentionally stricter
    than trusting the persisted workspace path on two axes:

    * **Out of jail** — a stale row or symlink must never turn session
      deletion into an out-of-jail ``rmtree``.
    * **In jail but too high** — ``workspace_volume_path`` is user-supplied
      via ``POST /v1/sessions`` and is NOT unique per session.  The account
      root ``<workspace_root>/<account_id>`` passes create-time validation
      (``is_relative_to`` is reflexive), so without a per-session ownership
      proof deleting one session would ``rmtree`` every OTHER live session's
      workspace for that tenant.  Refused here.

    Permitted workspace locations are the post-#409 canonical
    ``<workspace_root>/<account_id>/<session_id>`` and the pre-#409 legacy
    ``<workspace_root>/<session_id>`` (the same carve-out
    :func:`validate_workspace_path` makes, and still exclusively this
    session's).  Anything else — the account root, a sibling session's dir,
    a shared run/clone dir — is **skipped and logged**, never ``rmtree``d.

    Skipping is per-candidate, not all-or-nothing.  An unowned
    ``workspace_path`` does not suppress reclaiming this session's own
    uploads/attachments/repos dirs, which are derived from ``session_id``
    and are unambiguously its own; the earlier all-or-nothing shape leaked
    those forever for exactly the sessions with an anomalous workspace row.

    Refusing the ``rmtree`` is the safety property; aborting the *delete* is
    not.  This runs after ``delete_session`` has already committed the row
    removal, so raising here would report a failed DELETE for a session that
    is in fact gone — and would break the legitimate workflow-child case
    (``workspace='shared'`` children store the run's shared workspace).  Not
    deleting another session's live data is what matters; an unreferenced
    directory is recoverable by the reapers, another session's live
    workspace is not.
    """
    settings = get_settings()
    root = settings.workspace_root.resolve()
    workspace_bases = (
        settings.workspace_root / account_id / session_id,
        settings.workspace_root / session_id,
    )
    candidates = (
        (workspace_path, workspace_bases),
        (session_uploads_dir(session_id), (session_uploads_dir(session_id),)),
        (session_attachments_dir(session_id), (session_attachments_dir(session_id),)),
        (session_repos_root(session_id), (session_repos_root(session_id),)),
        (session_tmp_dir(session_id), (session_tmp_dir(session_id),)),
    )
    # Prove EVERY target before deleting ANY of them. A prove-as-you-go loop
    # would already have rmtree'd the earlier directories by the time a later
    # candidate turns out to be unowned; proving first keeps the destructive
    # phase free of any decision-making.
    live_paths = {Path(path).resolve() for path in live_workspace_paths}
    targets: list[Path] = []
    for candidate, bases in candidates:
        proven = _purge_target_if_owned(
            candidate, session_id=session_id, owned_bases=bases, root=root
        )
        if proven is None:
            continue
        if candidate == workspace_path and any(
            live == proven or live.is_relative_to(proven) for live in live_paths
        ):
            log.warning(
                "refusing to purge session workspace borrowed by a live session or run",
                path=str(candidate),
                session_id=session_id,
            )
            continue
        targets.append(proven)
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
