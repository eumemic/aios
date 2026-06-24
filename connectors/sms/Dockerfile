# SMS connector — runs in its own container, talks HTTP to aios (#1253).
#
# Unlike the outbound-only connectors, this sidecar ALSO stands up its own
# inbound webhook listener (the only inbound-reachable surface in the
# connector fleet — design §6). Expose the listener port and put it behind
# the operator's TLS-terminating public ingress (Traefik/Coolify) with a
# Twilio source-IP allowlist + rate limiting.
#
# Bind-mount ``<workspace_root>`` (host path) at the same path inside the
# container, read-only — outbound attachments (a later slice) arrive as
# ``/workspace/...`` strings the SDK resolves to host paths.

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
COPY connectors/sms /app/connectors/sms

# ``--no-sources`` tells uv to ignore each package's ``[tool.uv.sources]``
# block — those declare workspace-relative deps that only resolve from the
# repo root.  In this image we install the three packages by path in one
# invocation; uv satisfies the inter-package deps by name from the install
# set itself.
RUN uv pip install --system --no-cache --no-sources \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/sms

# Default listener port (overridable via AIOS_SMS_PORT).
EXPOSE 8080

CMD ["python", "-m", "aios_sms"]
