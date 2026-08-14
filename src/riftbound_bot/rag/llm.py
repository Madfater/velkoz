from __future__ import annotations

from langchain_openai import ChatOpenAI

# Shared with rag/vectorstore.py. The SDK's own defaults would let a stalled
# self-hosted or free gateway hang far longer than a Discord interaction can
# wait for.
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_MAX_RETRIES = 3


def build_chat_model(base_url: str, api_key: str, model: str, temperature: float = 0) -> ChatOpenAI:
    """Builds a ChatOpenAI client against an OpenAI-compatible endpoint.

    When no real key is configured (the free opencode.ai/zen gateway needs
    none), the openai SDK still requires api_key to be a non-empty string at
    construction time — but this specific gateway rejects any *non-empty*
    Authorization header with 401 "Invalid API key" (verified directly：a
    missing header or an empty bearer token both get 200, any real-looking
    token gets 401. So a placeholder satisfies the SDK's own validation,
    and default_headers overrides what's actually sent on the wire to
    nothing, only when no real key was configured.
    """
    keyless = not api_key
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key or "not-required",
        model=model,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
        **({"default_headers": {"Authorization": ""}} if keyless else {}),
    )
