# Reference HTTP-client connector — proves the deployment shape (#301).
#
# Real connectors (telegram, signal) follow the same pattern: thin
# python-slim base + only that connector's deps. The aios worker image
# stays free of connector-specific runtimes.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the SDK packages first so they're stable layers; then the connector.
# aios-connector-http depends on aios-sdk (the workspace-published typed SDK
# regenerated from openapi.json).
COPY packages/aios-sdk /app/packages/aios-sdk
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/echo-http /app/connectors/echo-http

RUN uv pip install --system --no-cache \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/echo-http

CMD ["python", "-m", "aios_echo_http"]
