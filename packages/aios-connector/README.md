# aios-connector

Reference SDK for building [aios](https://github.com/eumemic/aios)
connectors.  A connector is a stdio MCP subprocess aios spawns and
supervises.  It exposes:

- **Tools** (`signal_send`, `telegram_send`, etc.) the model calls.
- **Inbound notifications** (`notifications/aios/inbound`) carrying user
  messages from the underlying platform into aios sessions.
- **Account snapshots** (`notifications/aios/accounts`) the operator
  surfaces in `aios connectors list`.

## Quickstart

A connector subprocess can serve one account or many — pick whichever
shape matches your platform.  The SDK introspects each tool method's
signature and injects only what's declared.

### Single-account connector (one bot per process)

Use this shape for platforms where each account is a distinct process
identity (Telegram bots, Discord bots).  Operators deploy multiple
accounts by listing multiple instances under one connector type
(see "Multi-instance deployment" below).

```python
from typing import Any

from aios_connector import Connector, focal_required, make_account, tool


class MyConnector(Connector):
    name = "my_connector"

    async def discover_accounts(self) -> list[dict[str, Any]]:
        return [make_account(id="acct-1", display_name="My Bot")]

    @tool()
    @focal_required
    async def my_send(self, text: str, *, chat_id: str) -> dict[str, Any]:
        """Send `text` to the focal chat.  `chat_id` is injected from
        `_meta.aios.focal_channel_path`."""
        ...
```

### Multi-account connector (one process serves N accounts)

Use this shape for platforms whose daemon natively dispatches by
account (signal-cli, Matrix homeserver bridges).  `discover_accounts`
returns N entries; tool methods that target a specific chat declare
`account` in their signature so the SDK passes both account and chat
parts of the focal-channel path.

```python
class MyMultiConnector(Connector):
    name = "my_multi"

    async def discover_accounts(self) -> list[dict[str, Any]]:
        return [
            make_account(id="acct-1", display_name="Account One"),
            make_account(id="acct-2", display_name="Account Two"),
        ]

    @tool()
    @focal_required
    async def my_send(self, text: str, *, account: str, chat_id: str) -> dict[str, Any]:
        """Both `account` and `chat_id` are injected from
        `_meta.aios.focal_channel_path` (split on the first `/`)."""
        ...

    @tool()
    async def list_chats(self, account: str) -> list[dict[str, Any]]:
        """Non-focal tools that take `account` keep it as a model-visible
        argument — the model passes it explicitly per design §3.4."""
        ...
```

## Lifecycle hooks

The base class drives a fixed startup sequence; override the four
hooks to slot in your platform's bookkeeping.  Order is load-bearing —
inbounds emitted from `serve()` are blocked until the initial state has
been pushed, so `setup()` and `discover_accounts()` are guaranteed to
land before any inbound the model sees.

| Hook | When it runs | What's allowed |
|---|---|---|
| `setup()` | Before MCP server start | Open daemon connections, sign in, load credentials.  Blocking here keeps the supervisor in `starting` until either this returns or the supervisor's bounded init handshake (30s) trips and respawns. |
| `discover_accounts()` | Once, between `setup()` and server start | Return a list of account dicts (use `make_account`).  Most connectors return a one-element list for the single phone / bot the integrator owns. |
| `serve()` | Sibling task to the MCP server, after init | Long-running background work — typically a daemon listener that calls `emit_inbound`.  Override or accept the default (parks forever). |
| `teardown()` | Always, in `run()`'s `finally` | Close anything `setup()` opened.  Runs on normal exit, exception, and cancellation. |

Readiness deferral: the SDK does NOT send `notifications/initialized`
until `setup()` and `discover_accounts()` complete.  The supervisor's
`get_session()` blocks on first init; aios outbound dispatch into a
not-yet-ready connector waits, surfaces as a transport error after
60s.

## Account discovery

`discover_accounts()` returns a `list[dict[str, Any]]` — a list of
account snapshots, NOT a list of bare account-id strings.  Each entry
must include `id` and `display_name`; arbitrary metadata may be added.
Use the `make_account(id=..., display_name=..., metadata=...)`
convenience builder:

```python
async def discover_accounts(self) -> list[dict[str, Any]]:
    return [
        make_account(
            id=str(self._bot_id),
            display_name=self._first_name,
            metadata={"username": self._username},
        )
    ]
```

The `id` field is what aios uses as the opaque `account` segment in the
channel address `<connector>/<account>/<chat_id>`; it's the same value
your `emit_inbound(account=...)` calls must pass.  `display_name` and
`metadata` are surfaced in `aios connectors accounts <name>` and to
operator-facing CLI listings.

## Entry-point registration

Register the connector via the `aios.connectors` entry-point group in
your package's `pyproject.toml`.  The entry point resolves to a
zero-argument callable that returns a `Connector` instance:

```toml
[project.entry-points."aios.connectors"]
my_connector = "my_package:make_connector"
```

```python
# my_package/__init__.py
from .config import Settings
from .connector import MyConnector


def make_connector() -> MyConnector:
    return MyConnector(Settings())
```

That's it.  The supervisor synthesizes the launch command itself
(`python -m aios_connector run my_connector`), and the SDK's runner
loads the entry point, calls `make_connector()`, configures logging
via `AIOS_<MY_CONNECTOR>_LOG_FORMAT` / `_LOG_LEVEL`, and parks on
`Connector.run` until stdin EOF.

The supervisor stamps `cwd` to `settings.connectors_dir / connector`
(default instance) or `settings.connectors_dir / connector / instance`
(non-default), and creates the directory at boot so first-run
`FileNotFoundError` is impossible.

See `packages/aios-echo` for the canonical example.

## Multi-instance deployment

Operators run multiple instances of one connector type by listing them
in `connectors_enabled` with a `<connector>:<instance>` syntax:

```bash
AIOS_CONNECTORS_ENABLED=telegram:support,telegram:alerts,signal:main
AIOS_TELEGRAM_SUPPORT_BOT_TOKEN=xxx     # → AIOS_TELEGRAM_BOT_TOKEN inside support
AIOS_TELEGRAM_ALERTS_BOT_TOKEN=yyy      # → AIOS_TELEGRAM_BOT_TOKEN inside alerts
AIOS_SIGNAL_MAIN_PHONES=+1555,+1666     # → AIOS_SIGNAL_PHONES inside main
AIOS_SIGNAL_MAIN_CONFIG_DIR=/var/aios/signal
```

The supervisor re-exports `AIOS_<CONNECTOR>_<INSTANCE>_*` env vars as
`AIOS_<CONNECTOR>_*` inside each subprocess so connector code reads
its config under the standard prefix and stays instance-naive.  When
`<instance>` is omitted (e.g. `connectors_enabled=telegram`), the
default-instance shape applies: instance defaults to the connector
name, env vars are inherited unscoped, and cwd is the single-segment
`<connectors_dir>/<connector>/`.

Instance names match `^[a-z][a-z0-9_]*$` so the env-var re-export
stays POSIX-valid.

## Durability

`emit_inbound` writes to a SQLite spool at
`~/.aios/connectors/<name>/spool.sqlite` BEFORE pushing the
notification.  On reconnect the SDK replays unacked entries; the
worker-side dedup ledger (`connector_inbound_acks`) guarantees
at-most-once event append even across worker SIGKILL between commit
and ack.

## Attachments (inbound)

Pass a list of `Attachment` instances to `emit_inbound(attachments=...)`
when the platform delivers a photo, voice note, or document.  The
connector hands aios a *host path* it owns (e.g. signal-cli's
download dir, Telegram's `getFile()` result) — bytes never traverse
stdio.

```python
from aios_connector import Attachment

await self.emit_inbound(
    account=...,
    chat_id=...,
    sender=...,
    content="check this out",
    attachments=[
        Attachment(
            host_path="/tmp/signal-cli-downloads/abc.jpg",
            filename="photo.jpg",
            content_type="image/jpeg",
        ),
    ],
)
```

The SDK validates each attachment at `emit_inbound` time:

- `host_path` must be a regular file the SDK can stat.
- size must be ≤ 5 MiB.

Failures raise `AttachmentError` synchronously; the connector decides
whether to skip the attachment, send a placeholder text message, or
refuse the inbound.

The supervisor stages each attachment into a stable per-session
location and bind-mounts it read-only into the sandbox at
`/mnt/attachments/<connector>/<event-ulid>-<filename>`.

## Outbound attachments

Send tools take an `attachments: list[SandboxPath] | None` parameter
alongside `text`.  The model sees a JSON-schema `array of string` and
passes in-sandbox paths (`/workspace/foo.png`, `/mnt/attachments/...`);
the SDK resolves each to a host `pathlib.Path` before the tool body
runs.  No manual resolver call needed.

```python
from pathlib import Path
from typing import Any

from aios_connector import Connector, SandboxPath, focal_required, tool

class MyConnector(Connector):
    name = "my_connector"

    @tool()
    @focal_required
    async def my_send(
        self,
        text: str,
        attachments: list[SandboxPath] | None = None,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Send a message with optional attachments."""
        host_paths: list[Path] = list(attachments or [])
        # ... call platform send API with text + host_paths
        return {"status": "ok"}
```

Path allowlist: `/workspace/...` and `/mnt/attachments/...` only.
`/mnt/attachments/` is read-only inside the sandbox, so the model
must `cp` an inbound file into `/workspace/` before forwarding it.
The SDK rejects paths outside the allowlist (and `..` escapes) with a
`ValueError` before the tool body runs.
