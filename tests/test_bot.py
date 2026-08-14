import asyncio
import time
from types import SimpleNamespace

import discord
import httpx
from discord import app_commands
from openai import RateLimitError

from riftbound_bot.bot import (
    EMBED_DESCRIPTION_LIMIT,
    EMBED_FIELD_LIMIT,
    EMPTY_ANSWER_REPLY,
    MAX_HISTORY_TURNS,
    ThreadSession,
    _answer_embed,
    build_client,
)
from riftbound_bot.config import Settings
from riftbound_bot.rag.chain import RagResult


class FakeVectorstore:
    """No-op stand-in so build_client()'s RiftboundRagChain construction
    doesn't need a reachable embedding endpoint or an on-disk index — these
    tests exercise Discord glue (command registration, thread handling,
    error formatting), not RAG retrieval (see test_chain.py for that)."""

    def similarity_search(self, query, k, filter):
        return []


class FakeLLM:
    """Tripwire, not a stub: build_client constructs a real
    RiftboundRagChain around it before the tests swap in FakeChain, so this
    firing would mean a test is reaching the generation endpoint."""

    def invoke(self, messages):
        raise AssertionError("the real chain should never be invoked in these tests")


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "http://localhost/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


class FakeChain:
    def ask(self, question, history):
        return RagResult(answer="答案", citations_markdown="")


class RaisingChain:
    def __init__(self, error):
        self._error = error

    def ask(self, question, history):
        raise self._error


class SlowChain:
    """Records the history each call saw, and takes long enough that a
    second concurrent call would overlap it if nothing serialised them."""

    def __init__(self):
        self.histories = []

    def ask(self, question, history):
        self.histories.append(list(history))
        time.sleep(0.05)
        return RagResult(answer=f"回答：{question}", citations_markdown="")


class FakeResponse:
    def __init__(self):
        self.done = False

    async def defer(self, thinking=True):
        self.done = True

    async def send_message(self, content=None, ephemeral=False):
        self.done = True

    def is_done(self):
        return self.done


class FakeFollowup:
    def __init__(self, message=None):
        self.sent = []
        self._message = message

    async def send(
        self, content=None, embed=None, ephemeral=False, wait=False, allowed_mentions=None
    ):
        self.sent.append((content, embed, ephemeral))
        self.allowed_mentions = allowed_mentions
        # wait=True is what makes the real API return the created message;
        # the thread has to be anchored to that, not to defer()'s placeholder.
        return self._message if wait else None


class FakeThread:
    def __init__(self, id):
        self.id = id


class FakeMessage:
    def __init__(self, thread=None, create_thread_error=None):
        self._thread = thread
        self._error = create_thread_error
        self.created_thread_name = None

    async def create_thread(self, name, auto_archive_duration):
        if self._error is not None:
            raise self._error
        self.created_thread_name = name
        return self._thread


class FakeInteraction:
    def __init__(self, message):
        self.response = FakeResponse()
        self.followup = FakeFollowup(message)
        self.channel_id = 999
        self._message = message

    async def original_response(self):
        raise AssertionError(
            "original_response() returns defer()'s placeholder, not the answer message"
        )


class _FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeThreadChannel(discord.Thread):
    """A discord.Thread stand-in: subclassed (not instantiated) so
    isinstance(channel, discord.Thread) holds without discord.py's heavy
    gateway-payload-driven __init__.
    """

    def __init__(self, id, owner_id=None, typing_error=None):
        self.id = id
        self.owner_id = owner_id
        self.sent = []
        self._typing_error = typing_error

    def typing(self):
        if self._typing_error is not None:
            raise self._typing_error
        return _FakeTyping()

    async def send(self, content=None, embed=None, allowed_mentions=None):
        self.sent.append((content, embed))


class FakeAuthor:
    bot = False


class FakeThreadMessage:
    def __init__(self, channel, content="疾行是什麼？"):
        self.author = FakeAuthor()
        self.channel = channel
        self.content = content


class _FakeHTTPResponse:
    def __init__(self, status=403, reason="Forbidden"):
        self.status = status
        self.reason = reason


def _forbidden() -> discord.Forbidden:
    return discord.Forbidden(_FakeHTTPResponse(), {"code": 50001, "message": "Missing Access"})


def _build_client(tmp_path):
    settings = Settings(
        discord_bot_token="x",
        discord_guild_id=123,
        generation_base_url="http://localhost",
        generation_api_key="",
        generation_model="m",
        embedding_base_url="http://localhost",
        embedding_api_key="k",
        embedding_model="m",
        retrieval_pool_per_type=10,
        retrieval_k=6,
        retrieval_score_threshold=0.45,
        vector_store_dir=str(tmp_path / "turbovec"),
    )
    client = build_client(settings, vectorstore=FakeVectorstore(), llm=FakeLLM())
    client.chain = FakeChain()
    return client


def test_ask_registers_thread_history_when_thread_creation_succeeds(tmp_path):
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    interaction = FakeInteraction(FakeMessage(thread=FakeThread(id=42)))

    asyncio.run(ask.callback(interaction, "疾行是什麼？"))

    assert client.thread_sessions[42].history == [("human", "疾行是什麼？"), ("ai", "答案")]
    assert len(interaction.followup.sent) == 1


def test_ask_threads_off_the_answer_message_not_the_placeholder(tmp_path):
    # original_response() is defer()'s "thinking" message, so anchoring there
    # attached the conversation to a different message than the answer.
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    answer_message = FakeMessage(thread=FakeThread(id=42))
    interaction = FakeInteraction(answer_message)

    asyncio.run(ask.callback(interaction, "疾行是什麼？"))

    assert answer_message.created_thread_name == "疾行是什麼？"


def test_ask_does_not_ping_mentions_echoed_from_the_question(tmp_path):
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    interaction = FakeInteraction(FakeMessage(thread=FakeThread(id=42)))

    asyncio.run(ask.callback(interaction, "@everyone 疾行是什麼？"))

    mentions = interaction.followup.allowed_mentions
    assert (mentions.everyone, mentions.users, mentions.roles) == (False, False, False)


def test_ask_survives_forbidden_thread_creation(tmp_path):
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    interaction = FakeInteraction(FakeMessage(create_thread_error=_forbidden()))

    asyncio.run(ask.callback(interaction, "疾行是什麼？"))

    # The answer was still delivered, and no thread history was registered
    # for the thread that failed to be created.
    assert len(interaction.followup.sent) == 1
    assert client.thread_sessions == {}


def test_ask_survives_thread_creation_failing_for_any_other_reason(tmp_path):
    # /ask inside an existing thread raises HTTPException, not Forbidden —
    # which used to escape and report a failure for an answer already sent.
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    http_error = discord.HTTPException(_FakeHTTPResponse(400, "Bad Request"), "cannot nest")
    interaction = FakeInteraction(FakeMessage(create_thread_error=http_error))

    asyncio.run(ask.callback(interaction, "疾行是什麼？"))

    assert len(interaction.followup.sent) == 1
    assert client.thread_sessions == {}


def test_ask_error_reports_rate_limit_specific_message(tmp_path):
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    interaction = FakeInteraction(FakeMessage())
    asyncio.run(interaction.response.defer())
    wrapped = app_commands.CommandInvokeError(ask, _rate_limit_error())

    asyncio.run(ask.on_error(interaction, wrapped))

    assert interaction.followup.sent == [
        ("AI 服務目前使用量已達上限，請稍後再試一次。", None, True)
    ]


def test_on_message_reports_rate_limit_failure_and_skips_history(tmp_path):
    client = _build_client(tmp_path)
    client.chain = RaisingChain(_rate_limit_error())
    channel = FakeThreadChannel(id=42)
    client.thread_sessions[42] = ThreadSession()
    message = FakeThreadMessage(channel)

    asyncio.run(client.on_message(message))

    assert channel.sent == [("AI 服務目前使用量已達上限，請稍後再試一次。", None)]
    assert client.thread_sessions[42].history == []


def test_on_message_adopts_a_thread_this_bot_owns(tmp_path):
    # Sessions are in-memory, so after a restart every existing /ask thread
    # is untracked — without adoption the bot ignores them forever.
    client = _build_client(tmp_path)
    client._connection.user = SimpleNamespace(id=7)
    channel = FakeThreadChannel(id=42, owner_id=7)

    asyncio.run(client.on_message(FakeThreadMessage(channel)))

    assert client.thread_sessions[42].history == [("human", "疾行是什麼？"), ("ai", "答案")]
    assert len(channel.sent) == 1


def test_on_message_ignores_threads_the_bot_does_not_own(tmp_path):
    client = _build_client(tmp_path)
    client._connection.user = SimpleNamespace(id=7)
    channel = FakeThreadChannel(id=42, owner_id=99)

    asyncio.run(client.on_message(FakeThreadMessage(channel)))

    assert client.thread_sessions == {}
    assert channel.sent == []


def test_a_discord_side_failure_is_not_reported_as_an_ai_failure(tmp_path):
    # The typing indicator lives inside the same try; a permissions error
    # there used to be shown to the user as "AI 服務…錯誤".
    client = _build_client(tmp_path)
    channel = FakeThreadChannel(id=42, typing_error=_forbidden())
    client.thread_sessions[42] = ThreadSession()

    asyncio.run(client.on_message(FakeThreadMessage(channel)))

    assert channel.sent == []


def test_concurrent_followups_in_one_thread_are_serialised(tmp_path):
    # Two messages arriving during a chain call would otherwise both read the
    # same history and append in completion order.
    client = _build_client(tmp_path)
    client.chain = SlowChain()
    channel = FakeThreadChannel(id=42)
    client.thread_sessions[42] = ThreadSession()

    async def both():
        await asyncio.gather(
            client.on_message(FakeThreadMessage(channel, content="第一個問題")),
            client.on_message(FakeThreadMessage(channel, content="第二個問題")),
        )

    asyncio.run(both())

    history = client.thread_sessions[42].history
    assert [role for role, _ in history] == ["human", "ai", "human", "ai"]
    # The second question saw the first exchange rather than an empty history.
    assert client.chain.histories[1] == history[:2]


def test_thread_history_is_capped(tmp_path):
    client = _build_client(tmp_path)
    channel = FakeThreadChannel(id=42)
    client.thread_sessions[42] = ThreadSession()

    for _ in range(MAX_HISTORY_TURNS + 3):
        asyncio.run(client.on_message(FakeThreadMessage(channel)))

    assert len(client.thread_sessions[42].history) == MAX_HISTORY_TURNS * 2


def test_sessions_are_dropped_when_a_thread_is_deleted_or_archived(tmp_path):
    client = _build_client(tmp_path)
    client.thread_sessions[42] = ThreadSession()
    client.thread_sessions[43] = ThreadSession()

    asyncio.run(client.on_raw_thread_delete(SimpleNamespace(thread_id=42)))
    asyncio.run(
        client.on_thread_update(
            SimpleNamespace(id=43, archived=False), SimpleNamespace(id=43, archived=True)
        )
    )

    assert client.thread_sessions == {}


def test_an_over_long_answer_is_truncated_to_discords_limit():
    embed = _answer_embed(RagResult(answer="字" * 5000, citations_markdown="來" * 2000))

    assert len(embed.description) == EMBED_DESCRIPTION_LIMIT
    assert len(embed.fields[0].value) == EMBED_FIELD_LIMIT


def test_an_empty_answer_still_produces_a_valid_embed():
    # Discord rejects an empty description with a 400.
    embed = _answer_embed(RagResult(answer="", citations_markdown=""))

    assert embed.description == EMPTY_ANSWER_REPLY
