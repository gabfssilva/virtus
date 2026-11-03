from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from functools import reduce
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable
from uuid import uuid4

from virtus.filter import Capability, Filter, Next
from virtus.message import AssistantMessage, Event, Message, TextDelta, ToolCall, ToolResult, UserMessage
from virtus.request import Request
from virtus.tool import Tool

if TYPE_CHECKING:
    from virtus.agent import Agent


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

    async def send(self, message: UserMessage) -> object:
        """Run one turn from an already formatted user message and return the complete answer."""
        ...

    def child(self, key: tuple[str, str], create: Callable[[Callable[[], object]], Conversation]) -> Conversation:
        """Return the child under `key`, or register the one `create` builds from a detach callback."""
        ...


class Thread[**P, R](Capability, Conversation):
    """A conversation with an agent, kept across calls.

    The history holds what the provider actually received, after filters such as redaction or
    compaction, plus the replies, so those rewrites carry over to later turns. It is updated after
    each completed step: a turn rejected by a filter leaves it unchanged. Calls on the same thread
    run one at a time.

    Used as a capability, every delegation from any caller goes to this one thread.
    """

    def __init__(self, agent: Agent[P, R], id: str | None = None, detach: Callable[[], object] = lambda: None):
        self.agent: Final[Agent[P, R]] = agent
        self.id = id or uuid4().hex
        self._history: list[Message] = []
        self._children: dict[tuple[str, str], Conversation] = {}
        self._detach = detach
        self._disposed = False
        self._lock = asyncio.Lock()

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    @property
    def children(self) -> Mapping[tuple[str, str], Conversation]:
        return MappingProxyType(self._children)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.agent.output.consume(self._stream(self.agent.input(*args, **kwargs)))

    def dispose(self) -> None:
        """Forget the conversation and the threads opened from it, and unregister it from its agent.

        A disposed thread refuses further calls; get a fresh one from the agent instead.
        """
        for child in list(self._children.values()):
            child.dispose()
        self._history.clear()
        self._disposed = True
        self._detach()

    def filters(self) -> tuple[Filter, ...]:
        async def run(**arguments: object) -> object:
            return await self.send(self.agent.input(**arguments))

        return Tool(self.agent.name, self.agent.description, self.agent.parameters, run).filters()

    async def send(self, message: UserMessage) -> object:
        return await self.agent.output.final(self._stream(message))

    def child(self, key: tuple[str, str], create: Callable[[Callable[[], object]], Conversation]) -> Conversation:
        if found := self._children.get(key):
            return found

        created = create(lambda: self._children.pop(key, None))
        self._children[key] = created
        return created

    async def _stream(self, message: UserMessage) -> AsyncIterator[Event]:
        if self._disposed:
            raise RuntimeError(f"{self.agent.name}: thread {self.id!r} was disposed")

        async with self._lock:
            async for event in self._run(message):
                yield event

    async def _run(self, message: UserMessage) -> AsyncIterator[Event]:
        """Repeat steps until one ends without tool calls, updating the history after each step."""
        agent = self.agent
        sent: list[Request] = []
        pending = [*self._history, message]

        async def terminal(request: Request) -> AsyncIterator[Event]:
            async def execute(call: ToolCall) -> ToolResult:
                if found := next((t for t in request.tools if t.name == call.name), None):
                    return await found.execute(call)
                return ToolResult(call.id, call.name, f"Unknown tool: {call.name}", is_error=True)

            sent.append(request)
            reply: AssistantMessage | None = None

            async for event in request.provider.complete(request):
                if isinstance(event, AssistantMessage):
                    reply = event
                yield event

            if reply is None:
                raise RuntimeError(f"{request.provider!r} finished without an assistant message")

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(execute(call)) for call in reply.tool_calls]

            for task in tasks:
                yield task.result()

        def link(inner: Next, outer: Filter) -> Next:
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
                iteration=iteration,
                provider=agent.provider,
                thread=self,
            )

            async for event in step(request):
                if not isinstance(event, TextDelta):
                    produced.append(event)
                yield event

            pending = [*(sent[-1].messages if sent else pending), *produced]
            self._history[:] = pending

            match produced:
                case [*_, ToolResult()]:
                    continue
                case _:
                    return
