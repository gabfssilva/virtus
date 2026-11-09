from typing import TYPE_CHECKING

from virtus.providers.chat_completions import ChatCompletions
from virtus.providers.embeddings import Embeddings
from virtus.providers.factories import omlx, openrouter
from virtus.providers.responses import Responses

if TYPE_CHECKING:
    from virtus.providers.anthropic import Anthropic

__all__ = ["Anthropic", "ChatCompletions", "Embeddings", "Responses", "omlx", "openrouter"]


def __getattr__(name: str) -> object:
    """`Anthropic` loads on demand, so the `anthropic` extra is only needed by those who use it."""
    if name == "Anthropic":
        from virtus.providers.anthropic import Anthropic

        return Anthropic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
