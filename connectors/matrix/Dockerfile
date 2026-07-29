FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir uv
WORKDIR /app
# aios-matrix now depends on aios-connector-http (secrets-less runner
# declaration) which depends on aios-sdk; copy all three and install by
# path with --no-sources so the workspace-relative dep pins resolve from
# the install set itself (same pattern as the other connector images).
COPY packages/aios-sdk /app/packages/aios-sdk
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/matrix /app/connectors/matrix
RUN uv pip install --system --no-cache --no-sources \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/matrix
HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD ["python", "/app/connectors/matrix/healthcheck.py"]
CMD ["python", "-m", "aios_matrix"]
