# Changelog

All notable changes to `aios-sdk` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the SDK is in `0.x`, breaking changes may land at any minor bump.

## 0.1.0 — 2026-05-24

Initial public release.

- Typed Python SDK for the aios management plane, generated from
  `openapi.json` by `openapi-python-client`.
- Curated public surface: `Client`, `client_from_env`, `SseMessage`,
  `parse_sse_lines`, SSE streaming helpers (`stream_session`,
  `stream_connector_calls`, `stream_management_calls`,
  `stream_connection_discovery`), `UnexpectedStatus`. The full set is
  enumerated in `aios_sdk.__all__`.
- The bundled types match the FastAPI app at the commit this tag was
  cut from. Consumers should pin a deployment-compatible version; the
  SDK does not negotiate schema differences against an older runtime
  at install or call time.
