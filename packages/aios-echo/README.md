# aios-echo

Canonical example of an aios connector built on
[`aios-connector`](../aios-connector/).  Used in CI as the e2e fixture
exercising the supervisor's full discovery → spawn → init → tool-call
→ inbound → ack pipeline.

## Tools

- `ping` — returns `{"status": "pong"}`.
- `echo` — focal-required; echoes text tagged with the focal chat id.
- `trigger_inbound` — synthesizes an inbound message; the supervisor
  appends it to the bound session and acks the spool entry.

## Running

The supervisor finds this via the `aios.connectors` entry point group.
Add `echo` to your `connectors.enabled` config:

```yaml
connectors:
  enabled: [echo]
```

Then start `aios worker`; the supervisor spawns `python -m aios_echo`
and holds the persistent stdio MCP session.
