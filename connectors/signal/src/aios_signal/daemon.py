"""signal-cli subprocess lifecycle.

Spawns ``signal-cli daemon``, drains its stdio, waits for TCP readiness
via ``version``, and exposes :meth:`verify_phone` for the multi-connection
:class:`aios_signal.connector.SignalConnector` to validate each
connection's phone against ``accounts.json``.

Crash-is-fatal: an unexpected subprocess exit sets
:class:`DaemonCrashError` on :meth:`crashed`.  The connector's inbound
dispatcher (spawned under the runner's TaskGroup) distinguishes two
listener-drop cases via :meth:`subprocess_alive`:

* subprocess dead → fatal.  The drop's :class:`ListenerClosedError`
  propagates through the TaskGroup, tearing the container down so the
  operator restarts.
* subprocess alive (transient TCP blip) → the dispatcher reconnects the
  listener with bounded backoff instead of crashing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import structlog

from .addressing import encode_chat_id
from .errors import BotAccountNotFoundError, DaemonCrashError
from .rpc import RpcClient, RpcListener


@dataclass(frozen=True)
class GroupInfo:
    """One Signal group the bot is a member of.

    ``id`` is the URL-safe-base64 form used as the channel-path suffix
    (``signal/<bot>/<id>``).  ``member_uuids`` are the ACI UUIDs of the
    other participants; cross-reference with ``list_contacts`` for
    display names.
    """

    id: str
    name: str
    member_uuids: list[str]


log = structlog.get_logger(__name__)

READY_POLL_ATTEMPTS = 150  # 150 attempts @ 200ms = 30s total
READY_POLL_INTERVAL_S = 0.2
READY_POLL_TIMEOUT_S = 2.0

SHUTDOWN_GRACE_S = 5.0

daemon_exception_count = 0


class SignalDaemon:
    """Manages a single signal-cli daemon serving one or more accounts.

    Multi-account: launched without ``-a`` so signal-cli serves every
    registered account in ``config_dir/data/accounts.json``.  RPC calls
    must include ``account`` in their params; receive notifications
    carry ``params.account`` so callers know which phone the inbound
    arrived on (see :meth:`RpcListener.messages`).
    """

    def __init__(
        self,
        *,
        phones: list[str],
        config_dir: Path,
        cli_bin: str,
        host: str,
        port: int,
    ) -> None:
        self.phones = phones
        self.config_dir = config_dir
        self.cli_bin = cli_bin
        self.host = host
        self.port = port

        self._proc: asyncio.subprocess.Process | None = None
        self._drain_tasks: list[asyncio.Task[None]] = []
        self._watch_task: asyncio.Task[None] | None = None
        self._crash_future: asyncio.Future[None] | None = None

        self.rpc = RpcClient(host, port)
        self.listener = RpcListener(host, port)

    async def __aenter__(self) -> Self:
        await self._spawn()
        try:
            await self._wait_for_tcp()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None

        await self.listener.aclose()

        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=SHUTDOWN_GRACE_S)
            except TimeoutError:
                log.warning("signal.daemon.sigkill", phones=self.phones)
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
                await self._proc.wait()

        for t in self._drain_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._drain_tasks.clear()

    async def _spawn(self) -> None:
        # Multi-account daemon: no ``-a`` flag.  signal-cli will serve
        # every account registered in ``config_dir/data/accounts.json``;
        # callers route via ``account`` in RPC params.  Single-phone
        # setups still benefit from the same shape (one-element phones
        # list) — uniform code path beats a special case.
        args = [
            self.cli_bin,
            "--config",
            str(self.config_dir),
            "-o",
            "json",
            "--trust-new-identities",
            "always",
            "daemon",
            "--tcp",
            f"{self.host}:{self.port}",
            "--receive-mode=on-connection",
        ]
        log.info("signal.daemon.spawn", phones=self.phones, host=self.host, port=self.port)
        self._proc = await _spawn_subprocess(args)
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stdout, "signal.daemon.stdout"),
                name="signal-daemon-stdout",
            )
        )
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stderr, "signal.daemon.stderr"),
                name="signal-daemon-stderr",
            )
        )
        self._crash_future = asyncio.get_running_loop().create_future()
        self._watch_task = asyncio.create_task(self._watch_exit(), name="signal-daemon-watch")

    async def _watch_exit(self) -> None:
        assert self._proc is not None
        assert self._crash_future is not None
        rc = await self._proc.wait()
        if not self._crash_future.done():
            self._crash_future.set_exception(DaemonCrashError(f"signal-cli exited with code {rc}"))

    def crashed(self) -> asyncio.Future[None]:
        assert self._crash_future is not None, "daemon not started"
        return self._crash_future

    def subprocess_alive(self) -> bool:
        """True iff the signal-cli subprocess is still running.

        The inbound dispatcher consults this on a listener drop: alive
        means the TCP stream blipped while signal-cli kept running, so a
        reconnect is worth attempting; not-alive means the subprocess
        exited (its ``_watch_exit`` set :class:`DaemonCrashError` on the
        crash future) and the drop is fatal — let it propagate so the
        container restarts.
        """
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._crash_future is not None
            and not self._crash_future.done()
        )

    async def _wait_for_tcp(self) -> None:
        # ``version`` is the universal readiness probe — works in
        # multi-account daemon mode (no ``-a``) and is side-effect-free.
        last_error: Exception | None = None
        for _ in range(READY_POLL_ATTEMPTS):
            try:
                probe = RpcClient(self.host, self.port, timeout=READY_POLL_TIMEOUT_S)
                await probe.call("version")
                await self.listener.connect()
                log.info("signal.daemon.ready", host=self.host, port=self.port)
                return
            except Exception as e:
                last_error = e
                await asyncio.sleep(READY_POLL_INTERVAL_S)
        raise DaemonCrashError(
            f"signal-cli daemon never became ready on {self.host}:{self.port}: {last_error!r}"
        )

    def _read_accounts_index(self) -> dict[str, str]:
        """Parse signal-cli's on-disk ``accounts.json`` into ``{phone: uuid}``.

        Used by both :meth:`verify_phone` (single-phone lookup) and
        :meth:`discover_bot_uuids` (bulk lookup over ``self.phones``).
        Raises :class:`BotAccountNotFoundError` if the file is missing
        or malformed.
        """
        import json as _json

        accounts_json = self.config_dir / "data" / "accounts.json"
        try:
            raw = accounts_json.read_text()
        except OSError as e:
            raise BotAccountNotFoundError(
                f"cannot read signal-cli accounts file at {accounts_json}: {e}"
            ) from e
        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError as e:
            raise BotAccountNotFoundError(
                f"malformed signal-cli accounts file at {accounts_json}: {e}"
            ) from e
        registered: dict[str, str] = {}
        for entry in data.get("accounts", []):
            number = str(entry.get("number", "")).strip()
            uuid = entry.get("uuid")
            if number and isinstance(uuid, str) and uuid:
                registered[number] = uuid
        return registered

    async def verify_phone(self, phone: str) -> str:
        """Return the bot UUID for ``phone``, or raise.

        Per-phone counterpart to :meth:`discover_bot_uuids`. The
        multi-connection :class:`SignalConnector` calls this once per
        connection at ``serve_connection`` time; if the operator
        forgot to register the phone via
        ``signal-cli -a <phone> register``, the resulting
        :class:`BotAccountNotFoundError` propagates out and crashes
        the runtime container with the missing-account name in the
        traceback.
        """
        target = phone.strip()
        registered = self._read_accounts_index()
        uuid = registered.get(target)
        if uuid is None:
            accounts_json = self.config_dir / "data" / "accounts.json"
            raise BotAccountNotFoundError(
                f"signal-cli has no account for {target!r} in {accounts_json}. "
                f"Run `signal-cli -a {target} register` first."
            )
        return uuid

    async def discover_bot_uuids(self) -> dict[str, str]:
        """Return a ``{phone: uuid}`` map for every phone in ``self.phones``.

        Reads signal-cli's on-disk account index rather than RPCing —
        ``listAccounts`` works in multi-account daemon mode but
        accounts.json is the same source of truth and avoids a network
        round-trip during startup.

        Raises :class:`BotAccountNotFoundError` if any configured phone
        lacks a registered account.  Operators must register every
        phone in ``self.phones`` (via ``signal-cli -a <phone> register``)
        before launching the connector — surfacing the missing entry at
        startup beats discovering it on first inbound.
        """
        registered = self._read_accounts_index()
        out: dict[str, str] = {}
        missing: list[str] = []
        for phone in self.phones:
            target = phone.strip()
            uuid = registered.get(target)
            if uuid is None:
                missing.append(target)
            else:
                out[target] = uuid
        if missing:
            accounts_json = self.config_dir / "data" / "accounts.json"
            raise BotAccountNotFoundError(
                f"signal-cli has no account for {missing!r} in {accounts_json}. "
                f"Run `signal-cli -a <phone> register` for each missing number first."
            )
        return out

    async def list_groups(self, *, account: str) -> list[GroupInfo]:
        """Return the bot's group memberships via signal-cli ``listGroups``.

        Raises ``RpcError`` / ``RpcTimeoutError`` on RPC failure — the
        per-connection caller in
        :meth:`SignalConnector.serve_connection` uses the raise to
        refuse marking the account ready when signal-cli reports e.g.
        ``Specified account does not exist`` (the operator forgot to
        register it, or the registration expired).  The exception
        propagates out of the runner's discovery loop, tearing the
        container down so the operator sees red instead of green.

        Group IDs are re-encoded into URL-safe base64 so the caller
        can use them directly as channel-path suffixes
        (``signal/<bot>/<id>``) — matching :func:`encode_chat_id`'s
        ``group`` branch.  Groups missing either an ``id`` or
        ``members`` are dropped; the agent-facing roster is supposed
        to be a complete picture of "who is in this room with me."
        """
        result = await self.rpc.call("listGroups", {"account": account})
        if not isinstance(result, list):
            return []
        out: list[GroupInfo] = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("id")
            members = entry.get("members")
            if not isinstance(raw_id, str) or not raw_id:
                continue
            if not isinstance(members, list) or not members:
                continue
            name = entry.get("name") if isinstance(entry.get("name"), str) else ""
            member_uuids: list[str] = []
            for m in members:
                if not isinstance(m, dict):
                    continue
                uuid = m.get("uuid")
                if isinstance(uuid, str) and uuid:
                    member_uuids.append(uuid)
            if not member_uuids:
                continue
            out.append(
                GroupInfo(
                    id=encode_chat_id(raw_id, "group"),
                    name=name or "",
                    member_uuids=member_uuids,
                )
            )
        return out

    async def list_contacts(self, *, account: str) -> dict[str, str]:
        """Return a ``{uuid: display_name}`` map from signal-cli's contact store.

        signal-cli's inbound ``sourceName`` is sometimes empty (e.g. for peers
        not in the bot's local contacts but whose profile name Signal's UI
        resolves elsewhere). ``listContacts`` pulls the richer view so the
        connector can stamp a ``sender_name`` even when the envelope omits it.
        Returns an empty dict on RPC failure — name resolution is best-effort.
        """
        try:
            result = await self.rpc.call(
                "listContacts", {"account": account, "allRecipients": True}
            )
        except Exception:
            log.warning("signal.list_contacts.failed", account=account, exc_info=True)
            return {}
        if not isinstance(result, list):
            return {}
        mapping: dict[str, str] = {}
        for contact in result:
            if not isinstance(contact, dict):
                continue
            uuid = contact.get("uuid")
            if not isinstance(uuid, str) or not uuid:
                continue
            profile = contact.get("profile") or {}
            name = (
                contact.get("name")
                or (profile.get("givenName") if isinstance(profile, dict) else None)
                or (profile.get("familyName") if isinstance(profile, dict) else None)
            )
            if isinstance(name, str) and name.strip():
                mapping[uuid] = name.strip()
        return mapping

    async def register(self, *, phone: str, captcha: str | None, voice: bool) -> None:
        params: dict[str, Any] = {"account": phone, "voice": voice}
        if captcha is not None:
            params["captcha"] = captcha
        await self.rpc.call("register", params)

    async def verify(self, *, phone: str, code: str, pin: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {"account": phone, "verificationCode": code}
        if pin is not None:
            params["pin"] = pin
        result = await self.rpc.call("verify", params)
        return result if isinstance(result, dict) else {}

    async def update_profile(
        self,
        *,
        phone: str,
        given_name: str | None,
        family_name: str | None,
        about: str | None,
    ) -> None:
        # ``None`` is dropped — signal-cli treats absent params as
        # "no change" and null params as "clear field."
        params: dict[str, Any] = {"account": phone}
        if given_name is not None:
            params["givenName"] = given_name
        if family_name is not None:
            params["familyName"] = family_name
        if about is not None:
            params["about"] = about
        await self.rpc.call("updateProfile", params)


async def _spawn_subprocess(args: list[str]) -> asyncio.subprocess.Process:
    """Spawn signal-cli detached into its own session + process group.

    ``start_new_session=True`` (POSIX ``setsid()``) makes the child a
    session and process group leader.  Two payoffs:

    1. A foreground-terminal SIGINT (Ctrl-C) does not get forwarded to
       the daemon via the controlling terminal — the connector's own
       SIGINT handler runs ``__aexit__``, which terminates the daemon
       cleanly.  Without the new session, both the connector and the
       daemon get SIGINT simultaneously and the cleanup race is
       observable as half-shutdown state on the daemon's SQLite lock.
    2. Hard kills targeting the connector's pgroup (e.g. an operator
       running ``kill -TERM -<pgid>`` against the connector's group)
       no longer cascade to the daemon, so the daemon's controlled
       shutdown through ``__aexit__`` is not pre-empted.

    The corresponding shutdown asymmetry — that ``pkill -f aios_signal``
    only matches the Python process and not the daemon's JVM cmdline —
    is fixed by trapping SIGTERM in :func:`aios_signal.__main__.main`
    so ``__aexit__`` actually runs.
    """
    spawn = asyncio.create_subprocess_exec
    return await spawn(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


async def _drain(reader: asyncio.StreamReader, log_event: str) -> None:
    try:
        while True:
            line = await reader.readline()
            if not line:
                return
            decoded = line.rstrip(b"\n").decode("utf-8", errors="replace")
            log.info(log_event, line=decoded)
            if log_event == "signal.daemon.stdout":
                _log_daemon_exception(decoded)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning(f"{log_event}.read_error", error=str(e))


def _log_daemon_exception(line: str) -> None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict) or "exception" not in payload:
        return

    exception = payload["exception"]
    if not isinstance(exception, dict):
        return

    global daemon_exception_count
    daemon_exception_count += 1
    log.warning(
        "signal.daemon.exception",
        exception_type=exception.get("type"),
        exception_message=exception.get("message"),
        count=daemon_exception_count,
    )
