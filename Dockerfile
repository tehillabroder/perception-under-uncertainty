# keep the interpreter and installer fixed so rebuilds use the same starting point.
FROM ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 AS uv
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS base

FROM base AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
# the backend is locked too. no separate build environment resolves fresh versions.
RUN uv sync --locked --no-dev --group build --no-editable

FROM base AS runtime
ARG SOURCE_REVISION=unknown
ARG SOURCE_DIRTY=unknown
LABEL org.opencontainers.image.title="Perception Under Uncertainty" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PPU_SOURCE_REVISION="${SOURCE_REVISION}" \
    PPU_SOURCE_DIRTY="${SOURCE_DIRTY}" \
    PPU_LOCKFILE=/app/uv.lock \
    HOME=/tmp
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY uv.lock ./
COPY configs/ ./configs/
# a host bind mount needs its own writable permissions.. see the readme run command.
RUN useradd --create-home --uid 10001 --user-group --shell /usr/sbin/nologin ppu \
    && mkdir /results \
    && chown ppu:ppu /results
USER ppu
ENTRYPOINT ["ppu"]
CMD ["run", "--config", "configs/smoke.json", "--output-dir", "/results"]
