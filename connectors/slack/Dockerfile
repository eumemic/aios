# Slack connector — runs in its own container, talks HTTP to aios (#1229).
#
# Socket-Mode only: the sole inbound channel is the authenticated
# WebSocket the sidecar dials with the ``xapp-`` app token, so there is
# no public ingress, no signing secret, and no bind-mount needed in this
# slice (file upload — and its read-only ``<workspace_root>`` mount —
# is deferred).

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
COPY connectors/slack /app/connectors/slack

# ``--no-sources`` tells uv to ignore each package's ``[tool.uv.sources]``
# block — those declare workspace-relative deps (e.g.
# ``aios-sdk = { workspace = true }``) that only resolve from the
# repo root.  In this image we install the three packages by path in
# one invocation; uv satisfies the inter-package deps by name from
# the install set itself.
RUN uv pip install --system --no-cache --no-sources \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/slack

CMD ["python", "-m", "aios_slack"]
