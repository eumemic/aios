# aios-connector-http

Pure-HTTP-client SDK for building aios connectors. Supersedes the
MCP-stdio `aios-connector` SDK (deleted in PR 6 of the connector
rearchitecture, #301).

A connector is a process that:

1. Bridges an external chat platform (Telegram, Signal, Discord, …) to
   an aios connection's session(s).
2. POSTs inbound user messages to `POST /v1/connectors/inbound`.
3. Subscribes to `GET /v1/connectors/calls` (SSE) for pending custom
   tool calls referencing tools the connection declares; executes each
   tool by sending the right thing to the platform; POSTs the result
   to `POST /v1/sessions/:id/tool-results`.

The aios management API is the entire contract — connectors don't
share a process or a database with the worker.

## Quick start

```python
from aios_connector_http import HttpConnector, tool


class MyConnector(HttpConnector):
    @tool()
    async def my_send(self, *, chat_id: str, text: str) -> str:
        # send message via the platform; return the ack string the
        # session log will store.
        return "ok"


if __name__ == "__main__":
    import asyncio
    asyncio.run(MyConnector().run())
```

The connector reads `AIOS_URL` and `AIOS_CONNECTOR_TOKEN` from env to
locate and authenticate to the aios API. Platform credentials are read
from connector-specific env vars (e.g. `TELEGRAM_BOT_TOKEN`).
