"""Servers that speak Chat Completions, each a function that fills in `base_url` and the variable its key comes from.

Anything else compatible (DeepSeek, Ollama, Groq, Together, Gemini's OpenAI endpoint) is
`ChatCompletions(model, base_url=...)`.
"""

import os
from typing import TypedDict, Unpack

from openai.types.shared.reasoning_effort import ReasoningEffort

from virtus.providers.chat_completions import ChatCompletions


class Options(TypedDict, total=False):
    """The `ChatCompletions` options a compatible server takes too."""

    temperature: float | None
    max_tokens: int | None
    reasoning_effort: ReasoningEffort
    voice: str | None
    files: bool
    timeout: float
    max_retries: int


def openrouter(model: str, *, api_key: str | None = None, **options: Unpack[Options]) -> ChatCompletions:
    """OpenRouter; the key is `OPENROUTER_API_KEY` unless given. Models are named `vendor/model`."""
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    return ChatCompletions(model, api_key=key, base_url="https://openrouter.ai/api/v1", **options)


def omlx(
    model: str, *, api_key: str | None = None, base_url: str = "http://localhost:8123/v1", **options: Unpack[Options]
) -> ChatCompletions:
    """oMLX, serving MLX models on this machine; the key is `OMLX_API_KEY` if set. Files go as their text.

    The SDK insists on some key, and a server without authentication ignores it, so a placeholder stands in.
    """
    key = api_key or os.environ.get("OMLX_API_KEY") or "omlx"
    return ChatCompletions(model, api_key=key, base_url=base_url, **(Options(files=False) | options))
