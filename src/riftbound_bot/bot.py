"""Discord bot entrypoint.

/ask starts a RAG-grounded Q&A, replies inside a thread spawned from the bot's
own reply, and treats any further message posted in that thread as a follow-up
(with prior Q&A as context) — this is the "conversation" signal from the design
doc, no hand-rolled session tracking needed.

/card is the other half, and deliberately the opposite shape: an exact lookup
answered from stored card data with no retrieval and no LLM in the path, for
the questions that are just "what does this card do" and were being spent on
generation tokens.
"""
from __future__ import annotations

import asyncio

import discord
import structlog
from discord import app_commands
from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from psycopg_pool import ConnectionPool

from riftbound_bot.cards import (
    MAX_AUTOCOMPLETE_CHOICES,
    Card,
    CardCatalog,
    CardSource,
    PgCardSource,
)
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
MAX_EMBED_DESCRIPTION = 4096

# Riftbound's six domains plus colourless, as embed stripe colours. Keyed on
# the English colour word the card data stores, not the Chinese domain name.
_DOMAIN_EMBED_COLORS = {
    "red": 0xC0392B,
    "blue": 0x2E86C1,
    "green": 0x27916B,
    "purple": 0x7D3C98,
    "orange": 0xD35400,
    "yellow": 0xD4AC0D,
    "colorless": 0x95A5A6,
}

_DOMAIN_NAMES_ZH = {
    "red": "紅色",
    "blue": "藍色",
    "green": "綠色",
    "purple": "紫色",
    "orange": "橙色",
    "yellow": "黃色",
    "colorless": "無色",
}


class RiftboundClient(discord.Client):
    def __init__(
        self, settings: Settings, chain: RiftboundRagChain, catalog: CardCatalog
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.settings = settings
        self.chain = chain
        self.catalog = catalog
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


def _card_embed(card: Card) -> discord.Embed:
    """Renders one card as an embed: stats as fields, face image underneath.

    The image is `set_image` rather than `set_thumbnail` because on a TCG card
    the art *is* the card — it carries the printed stats and text that the
    fields above only transcribe. A thumbnail shrinks it to the point of being
    unreadable, which defeats the reason for showing it.
    """
    embed = discord.Embed(
        title=card.display_name,
        url=card.source_url or None,
        description=_card_description(card),
        color=_card_color(card),
    )
    for name, value in _card_stat_fields(card):
        embed.add_field(name=name, value=value, inline=True)
    if card.tags:
        embed.add_field(name="標籤", value="、".join(card.tags), inline=False)
    if card.image_url:
        embed.set_image(url=card.image_url)
    embed.set_footer(text=f"{card.id}｜{card.rarity}" if card.rarity else card.id)
    return embed


def _card_description(card: Card) -> str:
    # Discord rejects an embed whose description exceeds 4096 characters. No
    # Riftbound card comes close, but a truncated card beats a 400 from the API
    # if upstream ever ships a wall of reminder text.
    text = card.rules_text_zh.strip()
    return text[:MAX_EMBED_DESCRIPTION] if text else "（此卡無規則文字）"


def _card_color(card: Card) -> int:
    """Embed stripe matching the card's domain colour.

    Dual-domain cards (mostly Legends) carry "red/purple"; the stripe is one
    colour, so it takes the first — the domain the card is listed under.
    """
    primary = card.color.split("/")[0].strip().lower()
    return _DOMAIN_EMBED_COLORS.get(primary, EMBED_COLOR)


def _card_stat_fields(card: Card) -> list[tuple[str, str]]:
    """Only the stats this card actually has.

    energy/power/might are null far more often than not — a spell has no
    power, a battlefield has neither — and a field reading "力量：無" is noise
    dressed up as information.
    """
    fields = [("類型", card.category), ("顏色", _color_zh(card.color))]
    for label, value in (("費用", card.energy), ("力量", card.power), ("戰力", card.might)):
        if value is not None:
            fields.append((label, str(value)))
    return [(name, value) for name, value in fields if value]


def _color_zh(color: str) -> str:
    return "／".join(
        _DOMAIN_NAMES_ZH.get(part.strip().lower(), part) for part in color.split("/") if part
    )


def _connect_pool(settings: Settings) -> ConnectionPool:
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
    return pool


def _load_catalog(source: CardSource) -> CardCatalog:
    """Loads the card catalog, refusing card data too old to have images.

    Same up-front-failure argument as the index check above, and the same
    failure mode it guards against: `cards` rows written before the scraper
    captured `assets` carry no image_url, so /card would answer every lookup
    with a card-shaped embed and a conspicuous hole where the card should be.
    Better to refuse at boot, naming the command that fixes it, than to ship
    that to a channel.
    """
    catalog = CardCatalog.from_source(source)
    if not len(catalog):
        raise RuntimeError(
            "No cards in Postgres — run `python -m "
            "riftbound_bot.ingest.bootstrap` first."
        )
    if catalog.cards_missing_images:
        raise RuntimeError(
            f"{catalog.cards_missing_images} of {len(catalog)} cards have no "
            "image_url — this card data predates image capture. Re-run "
            "`python -m riftbound_bot.ingest.cards_scrape` to refresh it."
        )
    return catalog


def build_client(
    settings: Settings,
    vectorstore: RetrievalStore | None = None,
    llm: BaseChatModel | None = None,
    catalog: CardCatalog | None = None,
) -> RiftboundClient:
    """`vectorstore`/`llm`/`catalog` are injectable so tests can exercise the
    Discord command-registration/error-handling wiring here without needing a
    reachable database or embedding endpoint.
    """
    if vectorstore is None or catalog is None:
        # One pool serves both: the vector index the chain queries per request
        # and the single `cards` read the catalog makes at startup.
        pool = _connect_pool(settings)
        try:
            if vectorstore is None:
                vectorstore = PgVectorStore(
                    pool,
                    build_embeddings(
                        base_url=settings.embedding_base_url,
                        api_key=settings.embedding_api_key,
                        model=settings.embedding_model,
                    ),
                )
            if catalog is None:
                catalog = _load_catalog(PgCardSource(pool))
        except BaseException:
            pool.close()
            raise
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
    client = RiftboundClient(settings=settings, chain=chain, catalog=catalog)

    @client.tree.command(name="ask", description="詢問 Riftbound 規則或卡牌交互問題")
    @app_commands.describe(question="你的問題，例如：疾行是什麼意思？")
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

    @client.tree.command(name="card", description="以中英文卡名搜尋卡片資訊")
    @app_commands.describe(name="卡名或卡號，例如：巴凱旋沙者、Sandspinner、VEN-001")
    @app_commands.checks.cooldown(1, 3.0, key=lambda i: i.user.id)
    async def card(interaction: discord.Interaction, name: str) -> None:
        # No defer: the catalog is in memory and Discord renders the image from
        # the URL itself, so there is nothing slow to wait on. Deferring would
        # only add a visible "thinking…" flicker to an instant answer.
        found = client.catalog.resolve(name)
        if found is None:
            await interaction.response.send_message(
                f"找不到符合「{name}」的卡片。", ephemeral=True
            )
            return
        await interaction.response.send_message(embed=_card_embed(found))

    @card.autocomplete("name")
    async def card_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        """Offers matching cards as the user types, in both languages at once.

        The choice *value* is the card id, not the name: names are not unique,
        so submitting a name would throw away the disambiguation the user just
        performed by picking one row rather than another.
        """
        return [
            app_commands.Choice(name=found.choice_label(), value=found.id)
            for found in client.catalog.search(current, limit=MAX_AUTOCOMPLETE_CHOICES)
        ]

    @card.error
    async def card_error(
        interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f"查太快了，請再等 {error.retry_after:.0f} 秒。", ephemeral=True
            )
            return
        logger.exception("bot.card_failed", exc_info=error)
        message = "發生未預期的錯誤，請稍後再試一次。"
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
