# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cache layer independent of source changes).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

# uid/gid 1000 deliberately: ./data is bind-mounted from the host (see
# docker-compose.yml) and the mount carries the host's ownership, shadowing
# any chown done here. 1000 is the first non-system user on a typical
# single-user Linux host, so build_index can actually write the index it
# produces. Adjust both numbers if your host user differs.
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --home-dir /app app
WORKDIR /app
RUN mkdir -p /app/data && chown app:app /app/data

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src

# ./data (rules Markdown, the TurboVec index) is bind-mounted at runtime —
# see docker-compose.yml — so nothing under data/ is baked into the image;
# a build-time copy would always be shadowed by the mount and risk drifting
# stale against whatever's actually on the host.
# PYTHONUNBUFFERED so ingest progress appears as it happens: stdout is
# block-buffered when it isn't a TTY, which is exactly how the ingest
# containers run.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app

# No ENTRYPOINT: with `ENTRYPOINT ["python", "-m"]` the documented
# `docker compose run ingest python -m riftbound_bot.ingest.cards_scrape`
# expanded to `python -m python -m ...`, because `run` overrides the command,
# not the entrypoint. A plain CMD lets every documented invocation work.
CMD ["python", "-m", "riftbound_bot.bot"]
