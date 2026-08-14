import asyncio

import discord
import httpx
from discord import app_commands
from openai import RateLimitError

from riftbound_bot.bot import build_client
from riftbound_bot.config import Settings
from riftbound_bot.rag.chain import RagResult


class FakeVectorstore:
    """No-op stand-in so build_client()'s RiftboundRagChain construction
    doesn't need a reachable embedding endpoint or a populated index — these
    tests exercise Discord glue (command registration, thread handling,
    error formatting), not RAG retrieval (see test_chain.py for that)."""

    def fetch_by_source_type(self, source_type):
        return []


class FakeLLM:
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
    def __init__(self):
        self.sent = []

    async def send(self, content=None, embed=None, ephemeral=False):
        self.sent.append((content, embed, ephemeral))


class FakeThread:
    def __init__(self, id):
        self.id = id


class FakeMessage:
    def __init__(self, thread=None, create_thread_error=None):
        self._thread = thread
        self._error = create_thread_error

    async def create_thread(self, name, auto_archive_duration):
        if self._error is not None:
            raise self._error
        return self._thread


class FakeInteraction:
    def __init__(self, message):
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.channel_id = 999
        self._message = message

    async def original_response(self):
        return self._message


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

    def __init__(self, id):
        self.id = id
        self.sent = []

    def typing(self):
        return _FakeTyping()

    async def send(self, content=None, embed=None):
        self.sent.append((content, embed))


class FakeAuthor:
    bot = False


class FakeThreadMessage:
    def __init__(self, channel, content="疾行是什麼？"):
        self.author = FakeAuthor()
        self.channel = channel
        self.content = content


def _forbidden() -> discord.Forbidden:
    class _FakeHTTPResponse:
        status = 403
        reason = "Forbidden"

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
        database_url="postgresql://unused/unused",
    )
    client = build_client(settings, vectorstore=FakeVectorstore(), llm=FakeLLM())
    client.chain = FakeChain()
    return client


def test_ask_registers_thread_history_when_thread_creation_succeeds(tmp_path):
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    interaction = FakeInteraction(FakeMessage(thread=FakeThread(id=42)))

    asyncio.run(ask.callback(interaction, "疾行是什麼？"))

    assert client.thread_histories[42] == [("human", "疾行是什麼？"), ("ai", "答案")]
    assert len(interaction.followup.sent) == 1


def test_ask_survives_forbidden_thread_creation(tmp_path):
    client = _build_client(tmp_path)
    ask = client.tree.get_command("ask")
    interaction = FakeInteraction(FakeMessage(create_thread_error=_forbidden()))

    asyncio.run(ask.callback(interaction, "疾行是什麼？"))

    # The answer was still delivered, and no thread history was registered
    # for the thread that failed to be created.
    assert len(interaction.followup.sent) == 1
    assert client.thread_histories == {}


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
    client.thread_histories[42] = []
    message = FakeThreadMessage(channel)

    asyncio.run(client.on_message(message))

    assert channel.sent == [("AI 服務目前使用量已達上限，請稍後再試一次。", None)]
    assert client.thread_histories[42] == []
