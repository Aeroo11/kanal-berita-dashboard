# Serving image for the KANAL API.
#
# Targets Hugging Face Spaces, which runs the container as UID 1000 on a
# read-only filesystem apart from the home directory — hence the non-root user
# and the writable HOME. Getting that wrong produces a container that works
# locally and fails only once deployed, which is the worst place to find out.

FROM python:3.12-slim AS build

# uv, so the image resolves from the same lockfile the tests ran against. A
# `pip install` here would silently allow a different dependency tree in
# production than in CI.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies in their own layer, so a code change does not re-resolve them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

# HF Spaces runs containers as UID 1000. Creating the user explicitly means the
# same image behaves identically locally.
RUN useradd --create-home --uid 1000 kanal
USER kanal
ENV HOME=/home/kanal \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KANAL_REGISTRY=/app/registry

WORKDIR /app
COPY --from=build --chown=kanal:kanal /build/.venv /app/.venv
COPY --from=build --chown=kanal:kanal /build/src /app/src

# The registry travels with the image. An alias move inside a running container
# still takes effect without a restart — the loader re-reads it on a timer — but
# a rollback that must survive a container replacement belongs in a mounted
# volume, which is Stage 5's problem rather than this one's.
COPY --chown=kanal:kanal data/registry /app/registry

EXPOSE 7860

# Spaces expects 7860. One worker: the model is held per process, so a second
# worker would double the memory for no throughput this workload needs.
CMD ["uvicorn", "kanal.serving.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
