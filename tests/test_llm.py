"""Covers the keyless-gateway branch of chat-model construction.

The free opencode.ai/zen gateway 401s on any non-empty Authorization header
while the OpenAI SDK refuses to construct without a key, so the two have to
be satisfied separately. Nothing else in the codebase asserts that.
"""
from riftbound_bot.rag.llm import (
    REQUEST_MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    build_chat_model,
)


def test_a_keyless_gateway_gets_a_blank_authorization_header():
    llm = build_chat_model(base_url="http://gateway.test", api_key="", model="m")

    # The SDK needs a non-empty key; the wire must carry an empty header.
    assert llm.default_headers == {"Authorization": ""}


def test_a_real_key_is_sent_normally():
    llm = build_chat_model(base_url="http://gateway.test", api_key="sk-real", model="m")

    assert not llm.default_headers
    assert llm.openai_api_key.get_secret_value() == "sk-real"


def test_requests_are_bounded_in_both_modes():
    for api_key in ("", "sk-real"):
        llm = build_chat_model(base_url="http://gateway.test", api_key=api_key, model="m")
        assert llm.request_timeout == REQUEST_TIMEOUT_SECONDS
        assert llm.max_retries == REQUEST_MAX_RETRIES
