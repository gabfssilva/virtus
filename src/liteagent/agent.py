import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from functools import reduce
from itertools import count
from types import MappingProxyType
from typing import get_type_hints
from uuid import uuid4

from pydantic import Field, TypeAdapter, create_model

from liteagent.capability import flatten
from liteagent.message import AssistantMessage, Event, Message, TextDelta, ToolCall, ToolResult, UserMessage
from liteagent.provider import Provider
from liteagent.request import Capability, Conversation, Filter, Next, Request
from liteagent.tool import Tool, parameters_model


class Agent[**P, Out](Capability):
    """An LLM call loop described by a template function.

    The template's docstring, formatted with the call arguments, is the user message; its return
    annotation is the output type. Anything other than `str` is requested as structured output.

    Used as a capability, the agent becomes a tool other agents delegate to. The tool takes an
    optional `thread` name chosen by the calling model: reusing a name continues that conversation
    with this agent, omitting it makes a one-off request. These threads belong to the caller's
    thread, so separate conversations never share them.
    """

    def __init__(
        self,
        template: Callable[P, Awaitable[Out]],
        *,
        provider: Provider,
        name: str | None = None,
        description: str | None = None,
        system: str = "",
        capabilities: Sequence[Capability | Filter] = (),
    ) -> None:
        signature = inspect.signature(template)
        output = get_type_hints(template).get("return", str)

        self.name = name or template.__name__
        self.description = description or inspect.getdoc(template) or ""
        self.provider = provider
        self.system = system
        self._signature = signature
        self._prompt = inspect.getdoc(template) or "\n\n".join(f"{{{p}}}" for p in signature.parameters)
        self._filters = flatten(capabilities)
        self._parameters = parameters_model(template)
        self._output: TypeAdapter[Out] = TypeAdapter(output)
        self._schema = None if output is str else create_model("output", value=(output, ...)).model_json_schema()
        self._threads: dict[str, Thread[P, Out]] = {}

        if not self._prompt:
            raise TypeError(f"{self.name}: an agent needs parameters or a docstring")
        if "thread" in self._parameters.model_fields:
            raise TypeError(f"{self.name}: `thread` is reserved for delegation and can't be a parameter")

        inspect.markcoroutinefunction(self)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Out:
        return await Thread(self)._send(self._input(*args, **kwargs))

    def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncIterator[Event]:
        return Thread(self)._stream(self._input(*args, **kwargs))

    def thread(self, id: str | None = None) -> "Thread[P, Out]":
        """Return the thread with this id, creating it if needed; without an id, create a new one."""
        key = id or uuid4().hex
        if found := self._threads.get(key):
            return found

        created = Thread(self, key, detach=lambda: self._threads.pop(key, None))
        self._threads[key] = created
        return created

    def filters(self) -> tuple[Filter, ...]:
        parameters = create_model(
            self.name,
            __base__=self._parameters,
            thread=(
                str | None,
                Field(
                    None,
                    description=(
                        f"Name of a conversation with {self.name}. Reuse a name to continue that "
                        "conversation; omit it for a one-off request."
                    ),
                ),
            ),
        )

        async def delegate(request: Request, next: Next) -> AsyncIterator[Event]:
            parent = request.thread

            async def run(thread: str | None = None, **arguments: object) -> object:
                def create(detach: Callable[[], object]) -> Conversation:
                    return Thread(self, f"{parent.id}/{self.name}/{thread}", detach)

                target = parent._child((self.name, thread), create) if thread else Thread(self)
                return await target._send(self._input(**arguments))

            names = [name for agent, name in parent.children if agent == self.name]
            note = f"Open conversations with {self.name}: {', '.join(names)}" if names else ""
            system = "\n\n".join(part for part in (request.system, note) if part)
            tool = Tool(self.name, self.description, parameters, run)

            async for event in next(replace(request, system=system, tools=(*request.tools, tool))):
                yield event

        return (delegate,)

    def _input(self, *args: object, **kwargs: object) -> UserMessage:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return UserMessage(self._prompt.format(**bound.arguments))

    def _parse(self, content: str) -> Out:
        if self._schema is None:
            return self._output.validate_python(content)
        return self._output.validate_python(json.loads(content)["value"])

    async def _run(self, thread: "Thread[P, Out]", message: UserMessage) -> AsyncIterator[Event]:
        """Repeat steps until one ends without tool calls, updating the thread after each step."""
        sent: list[Request] = []
        pending = [*thread._history, message]

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
        step = reduce(link, reversed(self._filters), innermost)

        for iteration in count():
            sent.clear()
            produced: list[Message] = []
            request = Request(self.system, tuple(pending), (), self._schema, iteration, self.provider, thread)

            async for event in step(request):
                if not isinstance(event, TextDelta):
                    produced.append(event)
                yield event

            pending = [*(sent[-1].messages if sent else pending), *produced]
            thread._history[:] = pending

            match produced:
                case [*_, ToolResult()]:
                    continue
                case _:
                    return


class Thread[**P, Out](Capability, Conversation):
    """A conversation with an agent, kept across calls.

    The history holds what the provider actually received, after filters such as redaction or
    compaction, plus the replies, so those rewrites carry over to later turns. It is updated after
    each completed step: a turn rejected by a filter leaves it unchanged. Calls on the same thread
    run one at a time.

    Used as a capability, every delegation from any caller goes to this one thread.
    """

    def __init__(self, agent: Agent[P, Out], id: str | None = None, detach: Callable[[], object] = lambda: None):
        self.agent = agent
        self.id = id or uuid4().hex
        self._history: list[Message] = []
        self._children: dict[tuple[str, str], Conversation] = {}
        self._detach = detach
        self._lock = asyncio.Lock()

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    @property
    def children(self) -> Mapping[tuple[str, str], Conversation]:
        return MappingProxyType(self._children)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Out:
        return await self._send(self.agent._input(*args, **kwargs))

    def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncIterator[Event]:
        return self._stream(self.agent._input(*args, **kwargs))

    def dispose(self) -> None:
        """Forget the conversation and the threads opened from it, and unregister it from its agent."""
        for child in list(self._children.values()):
            child.dispose()
        self._history.clear()
        self._detach()

    def filters(self) -> tuple[Filter, ...]:
        async def run(**arguments: object) -> object:
            return await self._send(self.agent._input(**arguments))

        return Tool(self.agent.name, self.agent.description, self.agent._parameters, run).filters()

    async def _send(self, message: UserMessage) -> Out:
        async for _ in self._stream(message):
            pass

        match self._history[-1]:
            case AssistantMessage(content=content, tool_calls=()):
                return self.agent._parse(content)
            case last:
                raise RuntimeError(f"{self.agent.name}: the run ended without a final answer, last message: {last!r}")

    async def _stream(self, message: UserMessage) -> AsyncIterator[Event]:
        async with self._lock:
            async for event in self.agent._run(self, message):
                yield event

    def _child(self, key: tuple[str, str], create: Callable[[Callable[[], object]], Conversation]) -> Conversation:
        if found := self._children.get(key):
            return found

        created = create(lambda: self._children.pop(key, None))
        self._children[key] = created
        return created


def agent[**P, Out](
    *,
    provider: Provider,
    name: str | None = None,
    description: str | None = None,
    system: str = "",
    capabilities: Sequence[Capability | Filter] = (),
) -> Callable[[Callable[P, Awaitable[Out]]], Agent[P, Out]]:
    def decorate(template: Callable[P, Awaitable[Out]]) -> Agent[P, Out]:
        return Agent(
            template,
            provider=provider,
            name=name,
            description=description,
            system=system,
            capabilities=capabilities,
        )

    return decorate
