# Telegram connector — runs in its own container, talks HTTP to aios (#301).
#
# Bind-mount ``<workspace_root>`` (host path) at the same path inside the
# container, read-only — outbound attachments from the model arrive as
# ``/workspace/...`` strings the SDK resolves to host paths under
# ``<workspace_root>/<session_id>``.

# python:3.13-slim (pinned 2026-10-22)
FROM python:3.13-slim@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# uv binary (pinned).  Faster + reproducible vs ``pip install uv``.
# ghcr.io/astral-sh/uv:0.5.18 (pinned 2026-10-22)
COPY --from=ghcr.io/astral-sh/uv:0.5.18@sha256:e2101b9e627153b8fe4e8a1249cc4194f1b38ece7f28a5a9b8f958e3b560e69c /uv /usr/local/bin/uv

WORKDIR /app

# Lock-driven install, mirroring the api/worker image: ``--frozen`` means
# uv.lock alone decides every version — the build resolves nothing from
# PyPI.  ``--package aios-telegram`` scopes the sync to this member and its
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
COPY connectors/telegram/pyproject.toml ./connectors/telegram/
RUN uv sync --frozen --no-dev --package aios-telegram --no-install-workspace

COPY packages/aios-sdk ./packages/aios-sdk
COPY packages/aios-connector-http ./packages/aios-connector-http
COPY connectors/telegram ./connectors/telegram
RUN uv sync --frozen --no-dev --package aios-telegram --no-editable

# HEALTHCHECK / CMD invoke bare ``python`` — resolve it to the venv.
ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD ["python", "-m", "aios_connector_http.healthcheck"]
CMD ["python", "-m", "aios_telegram"]
