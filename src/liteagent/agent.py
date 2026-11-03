import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from functools import reduce
from itertools import count
from typing import get_type_hints

from pydantic import TypeAdapter, create_model

from liteagent.capability import flatten
from liteagent.message import AssistantMessage, Event, Message, TextDelta, ToolCall, ToolResult, UserMessage
from liteagent.provider import Provider
from liteagent.request import Capability, Filter, Next, Request
from liteagent.tool import Tool, parameters_model


class Agent[**P, Out](Capability):
    """An LLM call loop described by a template function.

    The template's docstring, formatted with the call arguments, is the user message; its return
    annotation is the output type. Anything other than `str` is requested as structured output.
    Used as a capability, the agent becomes a tool that other agents can delegate to.
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
        self._template = template
        self._signature = signature
        self._prompt = inspect.getdoc(template) or "\n\n".join(f"{{{p}}}" for p in signature.parameters)
        self._filters = flatten(capabilities)
        self._output: TypeAdapter[Out] = TypeAdapter(output)
        self._schema = None if output is str else create_model("output", value=(output, ...)).model_json_schema()

        if not self._prompt:
            raise TypeError(f"{self.name}: an agent needs parameters or a docstring")

        inspect.markcoroutinefunction(self)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Out:
        history: list[Message] = [self._input(*args, **kwargs)]
        async for _ in self._run(history):
            pass

        match history[-1]:
            case AssistantMessage(content=content, tool_calls=()):
                return self._parse(content)
            case last:
                raise RuntimeError(f"{self.name}: the run ended without a final answer, last message: {last!r}")

    def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncIterator[Event]:
        return self._run([self._input(*args, **kwargs)])

    def filters(self) -> tuple[Filter, ...]:
        return Tool(self.name, self.description, parameters_model(self._template), self).filters()

    def _input(self, *args: object, **kwargs: object) -> UserMessage:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return UserMessage(self._prompt.format(**bound.arguments))

    def _parse(self, content: str) -> Out:
        if self._schema is None:
            return self._output.validate_python(content)
        return self._output.validate_python(json.loads(content)["value"])

    async def _run(self, history: list[Message]) -> AsyncIterator[Event]:
        """Repeat steps until one ends without tool calls; `history` is replaced in place after each step."""
        sent: list[Request] = []

        async def terminal(request: Request) -> AsyncIterator[Event]:
            async def execute(call: ToolCall) -> ToolResult:
                if found := next((t for t in request.tools if t.name == call.name), None):
                    return await found.execute(call)
                return ToolResult(call.id, call.name, f"Unknown tool: {call.name}", is_error=True)

            sent.append(request)
            reply: AssistantMessage | None = None

            async for event in self.provider.complete(request):
                if isinstance(event, AssistantMessage):
                    reply = event
                yield event

            if reply is None:
                raise RuntimeError(f"{self.provider!r} finished without an assistant message")

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

            async for event in step(Request(self.system, tuple(history), (), self._schema, iteration)):
                if not isinstance(event, TextDelta):
                    produced.append(event)
                yield event

            history[:] = [*(sent[-1].messages if sent else history), *produced]

            match produced:
                case [*_, ToolResult()]:
                    continue
                case _:
                    return


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
