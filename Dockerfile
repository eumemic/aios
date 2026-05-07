# syntax=docker/dockerfile:1.6
#
# Multi-stage Dockerfile for aios. Two deploy targets share most of the
# image:
#
#   docker build --target api    -t aios-api    .
#   docker build --target worker -t aios-worker .
#
# Coolify points two Applications at this same Dockerfile with different
# `--target` values. When api/worker eventually split into separate repos,
# each takes the `base` + its own stage into a slimmer per-repo Dockerfile;
# the split is mechanical at that point.

# ── Stage 1: base — common deps + source ────────────────────────────────
FROM python:3.13-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# OS deps shared by both targets. curl is for the api healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

# uv binary (pinned). Faster + reproducible vs `pip install uv`.
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

# Non-root runtime user.
RUN useradd --create-home --uid 1000 aios

WORKDIR /app

# Install dependencies first for layer caching. Source comes after.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project source. Order: leaf-most-likely-to-change LAST.
COPY alembic.ini README.md ./
COPY migrations ./migrations
COPY connectors ./connectors
COPY packages ./packages
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# ── Stage 2: api ────────────────────────────────────────────────────────
# Stateless HTTP frontend. Doesn't talk to Docker; smallest surface area.
FROM base AS api

USER aios
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["aios", "api"]

# ── Stage 3: worker ─────────────────────────────────────────────────────
# Adds the docker CLI so the SandboxBackend can run/exec/rm sibling
# sandbox containers via the host docker daemon (mounted at /var/run/docker.sock
# at runtime). The daemon itself runs on the host — this is Docker-outside-of-
# Docker, not Docker-in-Docker.
FROM base AS worker

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && chmod a+r /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*

# Worker runs as root to access the bind-mounted docker socket without
# needing the host's docker GID baked into the image (which differs by host
# OS — 0 on Docker Desktop, 999/998 on most Linux). Inside the container,
# any UID can read /var/run/docker.sock when running as root. The sandbox
# containers it spawns run their own non-root UIDs.
USER root

CMD ["aios", "worker"]
