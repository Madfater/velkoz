# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cache layer independent of source changes).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
RUN mkdir -p /app/data && chown app:app /app/data

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src

# ./data (rules Markdown, the TurboVec index) is bind-mounted at runtime —
# see docker-compose.yml — so nothing under data/ is baked into the image;
# a build-time copy would always be shadowed by the mount and risk drifting
# stale against whatever's actually on the host.
ENV PATH="/app/.venv/bin:$PATH"
USER app

ENTRYPOINT ["python", "-m"]
CMD ["riftbound_bot.bot"]
