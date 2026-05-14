# Telegram connector — runs in its own container, talks HTTP to aios (#301).
#
# Bind-mount ``<workspace_root>`` (host path) at the same path inside the
# container, read-only — outbound attachments from the model arrive as
# ``/workspace/...`` strings the SDK resolves to host paths under
# ``<workspace_root>/<session_id>``.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the SDK packages first so the image-build cache stays warm across
# connector tweaks; then the connector package itself. aios-connector-http
# depends on aios-sdk (the workspace-published typed SDK).
COPY packages/aios-sdk /app/packages/aios-sdk
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/telegram /app/connectors/telegram

RUN uv pip install --system --no-cache \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/telegram

CMD ["python", "-m", "aios_telegram"]
