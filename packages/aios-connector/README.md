# aios-connector

Reference SDK for building [aios](https://github.com/eumemic/aios)
connectors.  A connector is a stdio MCP subprocess aios spawns and
supervises.  It exposes:

- **Tools** (`signal_send`, `telegram_send`, etc.) the model calls.
- **Inbound notifications** (`notifications/aios/inbound`) carrying user
  messages from the underlying platform into aios sessions.
- **Account snapshots** (`notifications/aios/accounts`) the operator
  surfaces in `aios connector list`.

## Quickstart

```python
from typing import Any

from aios_connector import Connector, focal_required, make_account, tool


class MyConnector(Connector):
    name = "my_connector"
    instructions = "Brief platform-specific guidance for the model."

    async def setup(self) -> None:
        # Open daemon connections, load credentials, etc.
        ...

    async def discover_accounts(self) -> list[dict[str, Any]]:
        return [make_account(id="acct-1", display_name="My Account")]

    @tool()
    @focal_required
    async def my_send(self, text: str, *, focal: str) -> dict[str, Any]:
        """Send `text` to the focal chat.

        `focal` is the chat-id; aios injects it from
        `_meta.aios.focal_channel_path` at dispatch time.
        """
        ...

    async def teardown(self) -> None:
        # Close daemon connections.
        ...
```

Register the connector via the `aios.connectors` entry-point group in
your package's `pyproject.toml`:

```toml
[project.entry-points."aios.connectors"]
my_connector = "my_package.factory:make_spec"
```

`make_spec(name, settings)` returns an
`aios.mcp.stdio_transport.ConnectorSpec` describing the launch command.
See `packages/aios-echo` for the canonical example.

## Durability

`emit_inbound` writes to a SQLite spool at
`~/.aios/connectors/<name>/spool.sqlite` BEFORE pushing the
notification.  On reconnect the SDK replays unacked entries; the
worker-side dedup ledger (`connector_inbound_acks`) guarantees
at-most-once event append even across worker SIGKILL between commit
and ack.
