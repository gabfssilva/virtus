from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from virtus.message import Event, Message, UserMessage

if TYPE_CHECKING:
    from virtus.provider import Provider
    from virtus.tool import Tool


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
    provider: Provider
    """The provider this step will call; filters may also use it directly, e.g. to summarize."""
    thread: Conversation
    """The conversation this request belongs to."""


type Next = Callable[[Request], AsyncIterator[Event]]
type Filter = Callable[[Request, Next], AsyncIterator[Event]]
"""Wraps one step of an agent run: a provider call followed by the tool calls it requested."""


@runtime_checkable
class Capability(Protocol):
    def filters(self) -> Sequence[Filter]: ...


@runtime_checkable
class Conversation(Protocol):
    """What filters and parent threads see of a thread, independent of its agent's types."""

    id: str

    @property
    def history(self) -> tuple[Message, ...]: ...

    @property
    def children(self) -> Mapping[tuple[str, str], Conversation]:
        """Threads opened by delegation from this one, keyed by (agent name, thread name)."""
        ...

    def dispose(self) -> None: ...

    async def _send(self, message: UserMessage) -> object: ...

    def _child(self, key: tuple[str, str], create: Callable[[Callable[[], object]], Conversation]) -> Conversation:
        """Return the child under `key`, or register the one `create` builds from a detach callback."""
        ...
