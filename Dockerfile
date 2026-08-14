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

COPY --from=builder --chown=app:app /app/.venv /app/.venv
# Carries src/riftbound_bot/ingest/seeds/ with it, which is the whole point of
# keeping those corpora inside the package: nothing is bind-mounted at runtime
# and the container writes no files, so there is no host directory to exist,
# own, or keep in sync. All data lives in Postgres.
COPY --chown=app:app src ./src

ENV PATH="/app/.venv/bin:$PATH"
USER app

ENTRYPOINT ["python", "-m"]
CMD ["riftbound_bot.bot"]
