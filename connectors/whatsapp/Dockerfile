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
# golang:1.25-alpine (pinned 2026-10-22)
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS daemon-build

WORKDIR /src

# Module download first so the layer cache survives daemon-source edits.
COPY connectors/whatsapp/daemon/go.mod connectors/whatsapp/daemon/go.sum ./
RUN go mod download

COPY connectors/whatsapp/daemon/ ./

# Static, CGo-disabled build — modernc.org/sqlite is pure-Go, so no
# libc dependency on the runtime side.  Multi-arch via buildx (the
# trailing $TARGETARCH is set by docker buildx; defaults to amd64
# under a plain ``docker build``).  No post-build ``-version`` smoke:
# under buildx with TARGETARCH != BUILDARCH the just-built binary
# isn't executable on the build host, and ``go build``'s exit code
# already proves the build succeeded.
ARG TARGETARCH=amd64
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build \
    -ldflags="-s -w -X main.Version=docker" \
    -o /out/whatsapp-daemon \
    ./cmd/whatsapp-daemon

# ─── Python runtime stage ────────────────────────────────────────────────
# python:3.13-slim-bookworm (pinned 2026-10-22)
FROM python:3.13-slim-bookworm@sha256:e32be0e55814a08977501ca863271d9bd02443710db8f2afaf8372e2c4508530 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIOS_WHATSAPP_DAEMON_BIN=/usr/local/bin/whatsapp-daemon \
    AIOS_WHATSAPP_DATA_DIR=/var/lib/aios/whatsapp-data

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv binary (pinned).  Faster + reproducible vs ``pip install uv``.
# ghcr.io/astral-sh/uv:0.5.18 (pinned 2026-10-22)
COPY --from=ghcr.io/astral-sh/uv:0.5.18@sha256:e2101b9e627153b8fe4e8a1249cc4194f1b38ece7f28a5a9b8f958e3b560e69c /uv /usr/local/bin/uv

# Drop the daemon binary in PATH.
COPY --from=daemon-build /out/whatsapp-daemon /usr/local/bin/whatsapp-daemon

WORKDIR /app

# Lock-driven install, mirroring the api/worker image: ``--frozen`` means
# uv.lock alone decides every version — the build resolves nothing from
# PyPI.  ``--package aios-whatsapp`` scopes the sync to this member and its
# workspace closure (aios-connector-http -> aios-sdk); the root ``aios``
# project (litellm & co.) is never installed.  uv reads the workspace from
# the root pyproject; members this image doesn't install are simply absent
# from /app, which the workspace member list tolerates (nothing matches).
#
# Deps first, from manifests only, so the third-party layer is cached until
# uv.lock moves; then the sources, installed ``--no-editable`` so the venv
# holds real wheels rather than .pth pointers into /app.
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
COPY packages/aios-sdk/pyproject.toml ./packages/aios-sdk/
COPY packages/aios-connector-http/pyproject.toml ./packages/aios-connector-http/
COPY connectors/whatsapp/pyproject.toml ./connectors/whatsapp/
RUN uv sync --frozen --no-dev --package aios-whatsapp --no-install-workspace

COPY packages/aios-sdk ./packages/aios-sdk
COPY packages/aios-connector-http ./packages/aios-connector-http
COPY connectors/whatsapp/README.md ./connectors/whatsapp/README.md
COPY connectors/whatsapp/src ./connectors/whatsapp/src
RUN uv sync --frozen --no-dev --package aios-whatsapp --no-editable

# entrypoint / HEALTHCHECK / CMD invoke bare ``python`` — resolve it to the venv.
ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint chowns the shared whatsapp-data dir so the api (uid 1000)
# can read media bytes the daemon decrypts there.  The daemon itself
# runs as root inside this container.
COPY connectors/whatsapp/entrypoint.sh /app/connectors/whatsapp/entrypoint.sh
RUN chmod 0755 /app/connectors/whatsapp/entrypoint.sh
HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD ["python", "-m", "aios_connector_http.healthcheck"]
ENTRYPOINT ["/app/connectors/whatsapp/entrypoint.sh"]
