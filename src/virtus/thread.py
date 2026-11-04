from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import reduce
from itertools import count
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable
from uuid import uuid4
from weakref import WeakValueDictionary

from virtus.journal import Journal, MemoryJournal
from virtus.live import Live
from virtus.message import (
    AssistantMessage,
    AudioDelta,
    Event,
    Message,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    UserMessage,
)
from virtus.middleware import Capability, Middleware, Next
from virtus.provider import LiveProvider, Session
from virtus.request import Request
from virtus.tool import Tool

if TYPE_CHECKING:
    from virtus.agent import Agent


@runtime_checkable
class Conversation(Protocol):
    """What middleware and parent threads see of a thread, independent of its agent's types."""

    id: str

    @property
    def path(self) -> tuple[str, ...]:
        """`(id,)` for a root thread; delegation appends (agent name, thread name) to the parent's path."""
        ...

    @property
    def history(self) -> tuple[Message, ...]: ...

    @property
    def children(self) -> Sequence[tuple[str, str]]:
        """(agent name, thread name) of the threads opened by delegation from this one."""
        ...

    def dispose(self) -> None: ...

    async def send(self, message: UserMessage) -> object:
        """Run one turn from an already formatted user message and return the complete answer."""
        ...

    def child(self, key: tuple[str, str], agent: Agent[..., object]) -> Conversation:
        """The thread under `key`, served by `agent`, in the same journal."""
        ...


class Thread[**P, R](Capability, Conversation):
    """A conversation with an agent, kept across calls.

    A thread is a handle: its state is the history in the journal, read before each run and written after
    each completed step. The history holds what the provider actually received, after middleware such as
    redaction or compaction, plus the replies, so those rewrites carry over to later turns; a turn rejected
    by a middleware leaves it unchanged. Handles to the same conversation in one process run one at a time.

    Without a journal of its own the thread gets a private `MemoryJournal`, which is what one-off calls
    use: nothing of theirs reaches the agent's journal.

    Used as a capability, every delegation from any caller goes to this one thread.
    """

    def __init__(self, agent: Agent[P, R], path: tuple[str, ...] | None = None, journal: Journal | None = None):
        self.agent: Final[Agent[P, R]] = agent
        self._path = path or (uuid4().hex,)
        self.id = "/".join(self._path)
        self._journal = journal or MemoryJournal()
        self._lock = _lock(self._journal, self._path)

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._journal.read(self._path) or ())

    @property
    def children(self) -> Sequence[tuple[str, str]]:
        return self._journal.children(self._path)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.agent.output.consume(self._stream(self.agent.input(*args, **kwargs)))

    def resume(self) -> R:
        """Finish a run that was interrupted, or return the last answer again if none was.

        A history ending in tool results continues from the next provider call; one ending in an
        answer replays that answer without calling anything. An empty thread has nothing to resume.
        """
        return self.agent.output.consume(self._stream(None))

    def dispose(self) -> None:
        """Delete the conversation and the threads opened from it from the journal."""
        self._journal.delete(self._path)

    @asynccontextmanager
    async def live(self) -> AsyncIterator[Live]:
        """Talk to this thread over the provider's realtime connection; see `Live`."""
        provider = self.agent.provider
        if not isinstance(provider, LiveProvider):
            raise TypeError(f"{provider!r} has no realtime connection")
        async with provider.connect() as session:
            yield Live(session, lambda message: self._stream(message, session))

    def middleware(self) -> tuple[Middleware, ...]:
        async def run(**arguments: object) -> object:
            return await self.send(self.agent.input(**arguments))

        return Tool(self.agent.name, self.agent.description, self.agent.parameters, run).middleware()

    async def send(self, message: UserMessage) -> object:
        return await self.agent.output.final(self._stream(message))

    def child(self, key: tuple[str, str], agent: Agent[..., object]) -> Conversation:
        return Thread(agent, (*self._path, *key), self._journal)

    async def _stream(self, message: UserMessage | None, live: Session | None = None) -> AsyncIterator[Event]:
        async with self._lock:
            async for event in self._run(message, live):
                yield event

    async def _run(self, message: UserMessage | None, live: Session | None) -> AsyncIterator[Event]:
        """Repeat steps until one ends without tool calls, updating the history after each step.

        Without a message, the run resumes from the history: nothing is called if it already ends in
        an answer, which is replayed instead. With a live session, it answers instead of the provider
        and gets the tool results back.
        """
        agent = self.agent
        sent: list[Request] = []
        pending = [*self.history, message] if message else [*self.history]

        match pending:
            case []:
                raise RuntimeError(f"{agent.name}: thread {self.id!r} has nothing to resume")
            case [*_, AssistantMessage(content=content, tool_calls=()) as answer] if message is None:
                yield TextDelta(content)
                yield answer
                return

        async def terminal(request: Request) -> AsyncIterator[Event]:
            async def execute(call: ToolCall) -> ToolResult:
                if found := next((t for t in request.tools if t.name == call.name), None):
                    return await found.execute(call)
                return ToolResult(call.id, call.name, f"Unknown tool: {call.name}", is_error=True)

            sent.append(request)
            reply: AssistantMessage | None = None
            events = live.respond(request.system, request.tools) if live else request.provider.complete(request)

            async for event in events:
                if isinstance(event, AssistantMessage):
                    reply = event
                yield event

            if reply is None:
                raise RuntimeError(f"{request.provider!r} finished without an assistant message")

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(execute(call)) for call in reply.tool_calls]

            for task in tasks:
                if live:
                    await live.deliver(task.result())
                yield task.result()

        def link(inner: Next, outer: Middleware) -> Next:
            return lambda request: outer(request, inner)

        innermost: Next = terminal
        step = reduce(link, reversed(agent.chain), innermost)

        for iteration in count():
            sent.clear()
            produced: list[Message] = []
            request = Request(
                system=agent.system,
                messages=tuple(pending),
                tools=(),
                output=agent.output.schema,
                media=("audio",) if live else agent.output.media,
                iteration=iteration,
                provider=agent.provider,
                thread=self,
            )

            async for event in step(request):
                if not isinstance(event, (TextDelta, AudioDelta, ToolCallDelta)):
                    produced.append(event)
                yield event

            pending = [*(sent[-1].messages if sent else pending), *produced]
            self._journal.write(self._path, pending)

            match produced:
                case [*_, ToolResult()]:
                    continue
                case _:
                    return


_locks: WeakValueDictionary[tuple[int, tuple[str, ...]], asyncio.Lock] = WeakValueDictionary()


def _lock(journal: Journal, path: tuple[str, ...]) -> asyncio.Lock:
    """One lock per conversation, shared by every handle to it and dropped with the last one."""
    key = (id(journal), path)
    if found := _locks.get(key):
        return found
    lock = _locks[key] = asyncio.Lock()
    return lock
