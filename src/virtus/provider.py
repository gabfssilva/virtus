from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from virtus.message import AssistantMessage, Delta, Image, ToolResult
from virtus.request import Request

if TYPE_CHECKING:
    from virtus.tool import Tool


@runtime_checkable
class Provider(Protocol):
    def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        """Stream the reply as text and tool call deltas, ending with exactly one complete `AssistantMessage`."""
        ...


type Kind = Literal["query", "document"]


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        """One vector per text, in order; `kind` says whether the texts are queries or the documents they look for."""
        ...


@runtime_checkable
class Session(Protocol):
    """An open realtime connection: the user's speech goes in, the model's comes out; see `Thread.live`.

    Audio is 24 kHz, 16-bit, mono PCM both ways. The client closes each turn with `commit`.
    """

    async def hear(self, pcm: bytes) -> None: ...

    async def show(self, image: Image) -> None: ...

    async def commit(self) -> None:
        """Close the user's turn: what was heard since the last commit is the next message."""
        ...

    def respond(self, system: str, tools: Sequence[Tool]) -> AsyncIterator[Delta | AssistantMessage]:
        """Ask for the reply to the turn, streamed as `complete` would, under this step's system prompt and tools."""
        ...

    async def deliver(self, result: ToolResult) -> None:
        """Hand a tool result back before the next `respond`."""
        ...

    async def interrupt(self, heard_ms: int) -> None:
        """Stop the reply in progress; the model keeps only the `heard_ms` the user actually heard."""
        ...


@runtime_checkable
class LiveProvider(Protocol):
    def connect(self) -> AbstractAsyncContextManager[Session]: ...
