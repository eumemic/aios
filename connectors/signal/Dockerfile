# Signal connector — runs in its own container, talks HTTP to aios (#301).
#
# signal-cli's daemon is launched as a subprocess (see daemon.py); the
# binary lives in this container, not in the worker image.  Per-account
# state (registered phones, contacts, groups) is in
# /home/aios/.signal-cli; bind-mount a host volume there to persist
# across rebuilds.
#
# Bind-mount ``<workspace_root>`` (host path) at the same path inside the
# container, read-only — outbound attachments from the model arrive as
# ``/workspace/...`` strings the SDK resolves to host paths under
# ``<workspace_root>/<session_id>``.

FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OS deps: ca-certificates for TLS + curl/tar for the signal-cli download.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

# uv binary (pinned).  Faster + reproducible vs `pip install uv`.
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

# signal-cli — upstream GraalVM native-image build.  Single self-contained
# ELF, no JRE required.  Pinned via ARG so version bumps are a one-line
# diff.  The trailing ``signal-cli --version`` is a build-time smoke
# test — a broken or glibc-incompatible binary fails the build instead
# of shipping quietly-broken.
#
# amd64-only: the upstream native tarball ships amd64 only.  Cross-arch
# local builds on Apple Silicon need ``--platform=linux/amd64``.
ARG SIGNAL_CLI_VERSION=0.14.3
RUN curl -fsSL "https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/signal-cli-${SIGNAL_CLI_VERSION}-Linux-native.tar.gz" \
        -o /tmp/signal-cli.tar.gz \
 && tar -xzf /tmp/signal-cli.tar.gz -C /usr/local/bin \
 && chmod 0755 /usr/local/bin/signal-cli \
 && rm /tmp/signal-cli.tar.gz \
 && signal-cli --version

WORKDIR /app

# SDK first so the layer cache stays warm across connector tweaks.
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/signal /app/connectors/signal

RUN uv pip install --system --no-cache \
    /app/packages/aios-connector-http \
    /app/connectors/signal

# Entrypoint chowns the shared state dir so the api (uid 1000) can read
# signal-cli's attachment writes from the same host bind mount.  Worker
# and connector containers run as root so the chown succeeds; the actual
# signal-cli process inside aios_signal still runs as root.
RUN chmod 0755 /app/connectors/signal/entrypoint.sh
ENTRYPOINT ["/app/connectors/signal/entrypoint.sh"]
