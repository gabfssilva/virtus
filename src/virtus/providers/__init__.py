from typing import TYPE_CHECKING

from virtus.providers.factories import omlx, openrouter
from virtus.providers.openai import OpenAI

if TYPE_CHECKING:
    from virtus.providers.anthropic import Anthropic

__all__ = ["Anthropic", "OpenAI", "omlx", "openrouter"]


def __getattr__(name: str) -> object:
    """`Anthropic` loads on demand, so the `anthropic` extra is only needed by those who use it."""
    if name == "Anthropic":
        from virtus.providers.anthropic import Anthropic

        return Anthropic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
