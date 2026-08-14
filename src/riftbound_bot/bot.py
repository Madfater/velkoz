"""Discord bot entrypoint: /ask starts a RAG-grounded Q&A, replies inside a
thread spawned from the bot's own reply, and treats any further message
posted in that thread as a follow-up (with prior Q&A as context) — this is
the "conversation" signal from the design doc, no hand-rolled session
tracking needed.

Per-thread state is in-memory and therefore disposable: the bot adopts any
thread it owns when a message arrives in one, so a restart costs the prior
turns of a conversation but never the ability to answer in it.
"""
from __future__ import annotations

import asyncio

import discord
import structlog
from discord import app_commands
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from openai import RateLimitError

from riftbound_bot.config import Settings
from riftbound_bot.logging_config import configure_logging
from riftbound_bot.rag.chain import RagResult, RiftboundRagChain
from riftbound_bot.rag.llm import build_chat_model
from riftbound_bot.rag.vectorstore import build_embeddings, load_vectorstore

logger = structlog.get_logger("riftbound_bot")

MAX_HISTORY_TURNS = 6
EMBED_COLOR = 0xC89B3C  # Riftbound-adjacent gold

# Discord's own payload limits. Exceeding any of them is a 400 that would
# surface as a generic command failure after the answer was already computed.
EMBED_DESCRIPTION_LIMIT = 4096
EMBED_FIELD_LIMIT = 1024
QUESTION_MAX_LENGTH = 500
THREAD_NAME_MAX_LENGTH = 90
THREAD_AUTO_ARCHIVE_MINUTES = 1440  # 24h

EMPTY_ANSWER_REPLY = "AI 沒有回覆任何內容，請再試一次。"


class ThreadSession:
    """One tracked Q&A thread: its history plus a lock serialising work on it.

    The lock matters because a thread is a conversation — two messages posted
    while a chain call is in flight would otherwise both read the same
    pre-existing history, answer without seeing each other, and append in
    whatever order they happened to finish.
    """

    def __init__(self, history: list[tuple[str, str]] | None = None) -> None:
        self.history: list[tuple[str, str]] = list(history or [])
        self.lock = asyncio.Lock()

    def record(self, question: str, answer: str) -> None:
        self.history.append(("human", question))
        self.history.append(("ai", answer))
        del self.history[: max(0, len(self.history) - MAX_HISTORY_TURNS * 2)]


class RiftboundClient(discord.Client):
    def __init__(self, settings: Settings, chain: RiftboundRagChain) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.settings = settings
        self.chain = chain
        self.tree = app_commands.CommandTree(self)
        # thread_id -> ThreadSession. Populated by /ask, and by adopting any
        # thread this bot owns (see _session_for), so histories survive the
        # restart that would otherwise orphan every existing thread.
        self.thread_sessions: dict[int, ThreadSession] = {}

    async def setup_hook(self) -> None:
        guild = discord.Object(id=self.settings.discord_guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.info("bot.commands_synced", guild_id=self.settings.discord_guild_id)

    async def on_ready(self) -> None:
        logger.info("bot.ready", user=str(self.user))

    def _session_for(self, thread: discord.Thread) -> ThreadSession | None:
        """The session for a thread, adopting bot-owned threads on sight.

        Sessions live in memory only, so without adoption a restart left every
        existing Q&A thread permanently unanswered — and a message posted in
        the gap between creating a thread and registering it was dropped too.
        An adopted thread starts with no history rather than none at all.
        """
        session = self.thread_sessions.get(thread.id)
        if session is not None:
            return session
        if self.user is not None and thread.owner_id == self.user.id:
            return self.thread_sessions.setdefault(thread.id, ThreadSession())
        return None

    def forget_thread(self, thread_id: int) -> None:
        self.thread_sessions.pop(thread_id, None)

    async def on_raw_thread_delete(self, payload: discord.RawThreadDeleteEvent) -> None:
        self.forget_thread(payload.thread_id)

    async def on_thread_update(self, before: discord.Thread, after: discord.Thread) -> None:
        # Archived threads are done; keeping their history would grow the map
        # for the lifetime of the process.
        if after.archived and not before.archived:
            self.forget_thread(after.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.Thread):
            return
        session = self._session_for(message.channel)
        if session is None:
            return  # not a thread this bot is tracking

        async with session.lock:
            try:
                async with message.channel.typing():
                    result = await self.run_chain(message.content, session.history)
            except discord.DiscordException:
                # A Discord-side failure (typing indicator, permissions) is
                # not the AI failing, and telling the user otherwise is
                # actively misleading. Nothing to reply through either.
                logger.exception("bot.thread_discord_error", channel_id=message.channel.id)
                return
            except Exception as error:
                logger.exception("bot.thread_followup_failed", channel_id=message.channel.id)
                await self._try_send(message.channel, content=_llm_failure_message(error))
                return

            session.record(message.content, result.answer)

        await self._try_send(message.channel, embed=_answer_embed(result))

    async def _try_send(self, channel: discord.abc.Messageable, **kwargs) -> None:
        """Best-effort send: the recovery path can't itself raise out of the
        event handler (an archived thread or a revoked permission would)."""
        try:
            await channel.send(allowed_mentions=discord.AllowedMentions.none(), **kwargs)
        except discord.DiscordException:
            logger.exception("bot.send_failed", channel_id=getattr(channel, "id", None))

    async def run_chain(self, question: str, history: list[tuple[str, str]]) -> RagResult:
        return await asyncio.to_thread(self.chain.ask, question, list(history))


def _llm_failure_message(error: BaseException) -> str:
    if isinstance(error, RateLimitError):
        return "AI 服務目前使用量已達上限，請稍後再試一次。"
    return "發生未預期的錯誤，請稍後再試一次。"


def _truncate(text: str, limit: int) -> str:
    """Trims to Discord's limit, marking that something was cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _answer_embed(result: RagResult) -> discord.Embed:
    # An empty description is a 400 from Discord, and the LLM returning
    # nothing is a real (if rare) outcome — an unhelpful reply beats a
    # command that appears to fail after the work was done.
    embed = discord.Embed(
        description=_truncate(result.answer or EMPTY_ANSWER_REPLY, EMBED_DESCRIPTION_LIMIT),
        color=EMBED_COLOR,
    )
    if result.citations_markdown:
        embed.add_field(
            name="來源",
            value=_truncate(result.citations_markdown, EMBED_FIELD_LIMIT),
            inline=False,
        )
    return embed


def build_client(
    settings: Settings,
    vectorstore: VectorStore | None = None,
    llm: BaseChatModel | None = None,
) -> RiftboundClient:
    """`vectorstore`/`llm` are injectable so tests can exercise the Discord
    command-registration/error-handling wiring here without needing a
    reachable embedding endpoint — RiftboundRagChain's constructor does a
    real embedding call even against an empty store (see chain.py's
    _load_card_docs_by_name), unlike the old Chroma-backed lookup this
    replaced, which was a pure metadata call with zero network I/O.
    """
    if vectorstore is None:
        embeddings = build_embeddings(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
        )
        vectorstore = load_vectorstore(settings.vector_store_dir, embeddings)
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
    async def ask(
        # Bounded because the question is echoed back inside a message, and
        # Discord would otherwise accept a 6000-character option into a
        # 2000-character message.
        interaction: discord.Interaction,
        question: app_commands.Range[str, 1, QUESTION_MAX_LENGTH],
    ) -> None:
        await interaction.response.defer(thinking=True)
        result = await client.run_chain(question, [])
        # wait=True to get the message back: original_response() returns the
        # "thinking" placeholder created by defer(), so threading off it hung
        # the conversation on a different message than the one carrying the
        # answer. The question is echoed verbatim, so mentions are disarmed.
        reply_message = await interaction.followup.send(
            content=f"**問題：** {question}",
            embed=_answer_embed(result),
            wait=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

        try:
            thread = await reply_message.create_thread(
                name=question[:THREAD_NAME_MAX_LENGTH] or "Riftbound 問答",
                auto_archive_duration=THREAD_AUTO_ARCHIVE_MINUTES,
            )
        except (discord.DiscordException, ValueError, TypeError):
            # The answer is already delivered, so nothing here should surface
            # as a command failure. Covers a missing "Create Public Threads"
            # permission, /ask used inside an existing thread (threads can't
            # nest), and forum channels.
            logger.warning(
                "bot.thread_create_failed", channel_id=interaction.channel_id, exc_info=True
            )
            return
        client.thread_sessions[thread.id] = ThreadSession(
            [("human", question), ("ai", result.answer)]
        )

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
    # log_handler=None: discord.py otherwise installs its own handler on the
    # 'discord' logger without clearing propagate, so every library log line
    # was emitted twice — once in discord's format, once as structlog JSON.
    client.run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()
