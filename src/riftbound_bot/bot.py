"""Discord bot entrypoint: /ask starts a RAG-grounded Q&A, replies inside a
thread spawned from the bot's own reply, and treats any further message
posted in that thread as a follow-up (with prior Q&A as context) — this is
the "conversation" signal from the design doc, no hand-rolled session
tracking needed.
"""
from __future__ import annotations

import asyncio

import discord
import structlog
from discord import app_commands
from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from psycopg_pool import ConnectionPool

from riftbound_bot.config import Settings
from riftbound_bot.logging_config import configure_logging
from riftbound_bot.rag.chain import RagResult, RiftboundRagChain
from riftbound_bot.rag.llm import build_chat_model
from riftbound_bot.rag.vectorstore import (
    PgVectorStore,
    RetrievalStore,
    build_embeddings,
    index_populated,
)

logger = structlog.get_logger("riftbound_bot")

MAX_HISTORY_TURNS = 6
EMBED_COLOR = 0xC89B3C  # Riftbound-adjacent gold


class RiftboundClient(discord.Client):
    def __init__(self, settings: Settings, chain: RiftboundRagChain) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.settings = settings
        self.chain = chain
        self.tree = app_commands.CommandTree(self)
        # thread_id -> alternating [("human", q), ("ai", a), ...], most recent last.
        self.thread_histories: dict[int, list[tuple[str, str]]] = {}

    async def setup_hook(self) -> None:
        guild = discord.Object(id=self.settings.discord_guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.info("bot.commands_synced", guild_id=self.settings.discord_guild_id)

    async def on_ready(self) -> None:
        logger.info("bot.ready", user=str(self.user))

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.Thread):
            return
        history = self.thread_histories.get(message.channel.id)
        if history is None:
            return  # not a thread this bot is tracking

        try:
            async with message.channel.typing():
                result = await self._run_chain(message.content, history)
        except Exception as error:
            logger.exception("bot.thread_followup_failed", channel_id=message.channel.id)
            await message.channel.send(_llm_failure_message(error))
            return

        history.append(("human", message.content))
        history.append(("ai", result.answer))
        del history[: max(0, len(history) - MAX_HISTORY_TURNS * 2)]

        await message.channel.send(embed=_answer_embed(result))

    async def _run_chain(self, question: str, history: list[tuple[str, str]]) -> RagResult:
        return await asyncio.to_thread(self.chain.ask, question, list(history))


def _llm_failure_message(error: BaseException) -> str:
    if isinstance(error, RateLimitError):
        return "AI 服務目前使用量已達上限，請稍後再試一次。"
    return "發生未預期的錯誤，請稍後再試一次。"


def _answer_embed(result: RagResult) -> discord.Embed:
    embed = discord.Embed(description=result.answer, color=EMBED_COLOR)
    if result.citations_markdown:
        embed.add_field(name="來源", value=result.citations_markdown, inline=False)
    return embed


def _connect_vectorstore(settings: Settings) -> PgVectorStore:
    """Opens the pool and refuses to serve an unbuilt index.

    The check is deliberately up front, at startup, rather than left to fail
    per-request: a bot answering every question from an empty index looks
    healthy while being uniformly wrong.
    """
    pool = ConnectionPool(settings.database_url, min_size=1, max_size=4, timeout=10)
    try:
        with pool.connection() as conn:
            if not index_populated(conn):
                raise RuntimeError(
                    "No vector index in Postgres — run `python -m "
                    "riftbound_bot.ingest.bootstrap` first."
                )
    except BaseException:
        # The pool starts worker threads on construction, so bailing out
        # without closing it leaks them and buries the actual startup error
        # under a "cannot join current thread" traceback from __del__.
        pool.close()
        raise
    embeddings = build_embeddings(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    return PgVectorStore(pool, embeddings)


def build_client(
    settings: Settings,
    vectorstore: RetrievalStore | None = None,
    llm: BaseChatModel | None = None,
) -> RiftboundClient:
    """`vectorstore`/`llm` are injectable so tests can exercise the Discord
    command-registration/error-handling wiring here without needing a
    reachable database or embedding endpoint.
    """
    if vectorstore is None:
        vectorstore = _connect_vectorstore(settings)
    if llm is None:
        llm = build_chat_model(
            base_url=settings.generation_base_url,
            api_key=settings.generation_api_key,
            model=settings.generation_model,
        )
    chain = RiftboundRagChain(
        vectorstore=vectorstore,
        llm=llm,
        pool_per_type=settings.retrieval_pool_per_type,
        k=settings.retrieval_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    client = RiftboundClient(settings=settings, chain=chain)

    @client.tree.command(name="ask", description="詢問 Riftbound 規則或卡牌交互問題")
    @app_commands.describe(question="你的問題，例如：暴怒屬性的疾行是什麼意思？")
    @app_commands.checks.cooldown(1, 10.0, key=lambda i: i.user.id)
    async def ask(interaction: discord.Interaction, question: str) -> None:
        await interaction.response.defer(thinking=True)
        result = await client._run_chain(question, [])
        await interaction.followup.send(
            content=f"**問題：** {question}", embed=_answer_embed(result)
        )

        reply_message = await interaction.original_response()
        try:
            thread = await reply_message.create_thread(
                name=question[:90] or "Riftbound 問答",
                auto_archive_duration=1440,
            )
        except discord.Forbidden:
            # Answer was already delivered above; a missing "Create Public
            # Threads" permission in this channel shouldn't surface as a
            # command failure — just skip follow-up thread tracking.
            logger.warning(
                "bot.thread_create_forbidden", channel_id=interaction.channel_id
            )
            return
        client.thread_histories[thread.id] = [("human", question), ("ai", result.answer)]

    @ask.error
    async def ask_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f"問太快了，請再等 {error.retry_after:.0f} 秒。", ephemeral=True
            )
            return
        logger.exception("bot.ask_failed", exc_info=error)
        original = error.original if isinstance(error, app_commands.CommandInvokeError) else error
        message = _llm_failure_message(original)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    return client


def main() -> None:
    configure_logging()
    settings = Settings.load()
    client = build_client(settings)
    client.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
