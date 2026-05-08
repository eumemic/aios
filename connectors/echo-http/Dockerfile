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

# Copy the SDK package first so it's a stable layer; then the connector.
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/echo-http /app/connectors/echo-http

RUN uv pip install --system --no-cache \
    /app/packages/aios-connector-http \
    /app/connectors/echo-http

CMD ["python", "-m", "aios_echo_http"]
