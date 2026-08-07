FROM python:3.12.11-slim@sha256:27f90d79cc85e9b7b2560063ef44fa0e9eaae7a7c3f5a9f74563065c5477cc24 AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY milvus_trace/docker/requirements-replay.lock /tmp/requirements-replay.lock
RUN --mount=type=secret,id=corp_ca,target=/tmp/corp-ca.crt,required=false \
    if [ -s /tmp/corp-ca.crt ]; then \
      mkdir -p /usr/local/share/ca-certificates && \
      cp /tmp/corp-ca.crt /usr/local/share/ca-certificates/corp-ca.crt && \
      update-ca-certificates; \
    fi && \
    pip install --no-cache-dir --target=/opt/replay-deps -r /tmp/requirements-replay.lock
FROM python:3.12.11-slim@sha256:27f90d79cc85e9b7b2560063ef44fa0e9eaae7a7c3f5a9f74563065c5477cc24
ENV PYTHONPATH=/opt/replay-deps PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=dependencies /opt/replay-deps /opt/replay-deps
COPY milvus_trace /app/milvus_trace
RUN useradd --create-home --uid 10001 replayer
USER replayer
ENTRYPOINT ["python", "-m", "milvus_trace.replay"]
