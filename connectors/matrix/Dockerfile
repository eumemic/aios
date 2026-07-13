FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY connectors/matrix /app/connectors/matrix
RUN uv pip install --system --no-cache /app/connectors/matrix
HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD ["python", "/app/connectors/matrix/healthcheck.py"]
CMD ["python", "-m", "aios_matrix"]
