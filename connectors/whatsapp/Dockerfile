# WhatsApp connector — runs in its own container, talks HTTP to aios.
#
# Two languages in one image: the whatsapp-daemon is a static Go binary
# built from connectors/whatsapp/daemon/, the Python connector is the
# usual aios_connector_http subclass.  The daemon is launched as a
# subprocess of the Python connector (see daemon.py); its sqlstore +
# inbound media land under /var/lib/aios/whatsapp-data/<phone>/.
# Bind-mount a host volume there to persist across rebuilds.
#
# Bind-mount ``<workspace_root>`` (host path) at the same path inside the
# container, read-only — outbound attachments from the model arrive as
# ``/workspace/...`` strings the SDK resolves to host paths under
# ``<workspace_root>/<session_id>``.

# ─── Go build stage ──────────────────────────────────────────────────────
FROM golang:1.25-alpine AS daemon-build

WORKDIR /src

# Module download first so the layer cache survives daemon-source edits.
COPY connectors/whatsapp/daemon/go.mod connectors/whatsapp/daemon/go.sum ./
RUN go mod download

COPY connectors/whatsapp/daemon/ ./

# Static, CGo-disabled build — modernc.org/sqlite is pure-Go, so no
# libc dependency on the runtime side.  Multi-arch via buildx (the
# trailing $TARGETARCH is set by docker buildx; defaults to amd64
# under a plain ``docker build``).
ARG TARGETARCH=amd64
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build \
    -ldflags="-s -w -X main.Version=docker" \
    -o /out/whatsapp-daemon \
    ./cmd/whatsapp-daemon \
 && /out/whatsapp-daemon -version

# ─── Python runtime stage ────────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIOS_WHATSAPP_DAEMON_BIN=/usr/local/bin/whatsapp-daemon \
    AIOS_WHATSAPP_DATA_DIR=/var/lib/aios/whatsapp-data

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv binary (pinned).  Faster + reproducible vs ``pip install uv``.
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

# Drop the daemon binary in PATH.
COPY --from=daemon-build /out/whatsapp-daemon /usr/local/bin/whatsapp-daemon

WORKDIR /app

# SDK packages first so the layer cache stays warm across connector tweaks.
# aios-connector-http depends on aios-sdk (the workspace-published typed SDK).
COPY packages/aios-sdk /app/packages/aios-sdk
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/whatsapp/pyproject.toml /app/connectors/whatsapp/pyproject.toml
COPY connectors/whatsapp/src /app/connectors/whatsapp/src

# ``--no-sources`` ignores ``[tool.uv.sources]`` workspace-deps which
# only resolve at the repo root; uv satisfies the inter-package deps
# by name from the install set itself.
RUN uv pip install --system --no-cache --no-sources \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/whatsapp

# Entrypoint chowns the shared whatsapp-data dir so the api (uid 1000)
# can read media bytes the daemon decrypts there.  The daemon itself
# runs as root inside this container.
COPY connectors/whatsapp/entrypoint.sh /app/connectors/whatsapp/entrypoint.sh
RUN chmod 0755 /app/connectors/whatsapp/entrypoint.sh
ENTRYPOINT ["/app/connectors/whatsapp/entrypoint.sh"]
