# aios-whatsapp

WhatsApp connector for aios.

Pairs a WhatsApp account (via the unofficial Multi-Device protocol, backed by
[`go.mau.fi/whatsmeow`](https://github.com/tulir/whatsmeow)) and surfaces messages
into an aios session. Mirrors the shape of `aios-signal`: the Python
`HttpConnector` spawns a Go daemon (`whatsapp-daemon`) as a subprocess and talks to
it over line-delimited JSON-RPC on a loopback TCP port.

This connector is **under construction**. Tracking: see `connectors/whatsapp/` in
the aios tree.
