from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from liteagent.message import Event, Message

if TYPE_CHECKING:
    from liteagent.tool import Tool


@dataclass(frozen=True)
class Request:
    """One call to the provider, as seen and rewritten by filters.

    The `messages` that reach the provider become the conversation history, so a filter that
    rewrites them (redaction, compaction) changes what is remembered, not only what is sent.
    """

    system: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    output: Mapping[str, object] | None
    """JSON schema the final answer must follow; `None` for plain text."""
    iteration: int
    """0 for the first provider call of a run, incremented after each round of tool calls."""


type Next = Callable[[Request], AsyncIterator[Event]]
type Filter = Callable[[Request, Next], AsyncIterator[Event]]
"""Wraps one step of an agent run: a provider call followed by the tool calls it requested."""


@runtime_checkable
class Capability(Protocol):
    def filters(self) -> Sequence[Filter]: ...
