import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import takewhile
from types import MethodType

from virtus.filter import Capability, Filter, Next
from virtus.message import Event, Message, TextDelta, UserMessage
from virtus.request import Request
from virtus.tool import Tool


class Capabilities(Capability):
    """Groups tools, context and hooks declared as methods, so they can share state through `self`.

    Tool names are prefixed with the class name in snake case: `Memory.remember` -> `memory__remember`.
    """

    def filters(self) -> tuple[Filter, ...]:
        prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
        names = dict.fromkeys(
            name
            for klass in reversed(type(self).__mro__)
            for name, value in vars(klass).items()
            if isinstance(value, Capability)
        )

        def member_filters(name: str) -> Iterator[Filter]:
            match getattr(self, name):
                case Tool() as tool:
                    yield from replace(tool, name=f"{prefix}__{tool.name}").filters()
                case Capability() as capability:
                    yield from capability.filters()

        return tuple(f for name in names for f in member_filters(name))


@dataclass(frozen=True)
class Context(Capability):
    function: Callable[..., str | Awaitable[str]]

    def __get__(self, instance: object | None, owner: type | None = None) -> "Context":
        return self if instance is None else replace(self, function=MethodType(self.function, instance))

    def filters(self) -> tuple[Filter, ...]:
        async def add(request: Request, next: Next) -> AsyncIterator[Event]:
            text = await _resolve(self.function())
            system = "\n\n".join(part for part in (request.system, text) if part)
            async for event in next(replace(request, system=system)):
                yield event

        return (add,)


@dataclass(frozen=True)
class OnInput(Capability):
    function: Callable[..., UserMessage | Awaitable[UserMessage]]

    def __get__(self, instance: object | None, owner: type | None = None) -> "OnInput":
        return self if instance is None else replace(self, function=MethodType(self.function, instance))

    def filters(self) -> tuple[Filter, ...]:
        async def rewrite(request: Request, next: Next) -> AsyncIterator[Event]:
            if request.iteration == 0:
                request = replace(request, messages=await self._rewrite(request.messages))

            async for event in next(request):
                yield event

        return (rewrite,)

    async def _rewrite(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        """The new input is the run of user messages at the end of the conversation."""
        start = len(messages) - sum(1 for _ in takewhile(lambda m: isinstance(m, UserMessage), reversed(messages)))
        rewritten = [await _resolve(self.function(m)) if isinstance(m, UserMessage) else m for m in messages[start:]]
        return (*messages[:start], *rewritten)


@dataclass(frozen=True)
class OnMessage(Capability):
    """Sees every complete message: assistant replies, including tool calls, and tool results.

    Streamed `TextDelta`s pass through untouched. Raising on an assistant message stops the run
    before its tool calls execute; returning a modified message does not change which calls run.
    """

    function: Callable[..., Message | Awaitable[Message]]

    def __get__(self, instance: object | None, owner: type | None = None) -> "OnMessage":
        return self if instance is None else replace(self, function=MethodType(self.function, instance))

    def filters(self) -> tuple[Filter, ...]:
        async def apply(request: Request, next: Next) -> AsyncIterator[Event]:
            async for event in next(request):
                match event:
                    case TextDelta():
                        yield event
                    case message:
                        result = await _resolve(self.function(message))
                        if type(result) is not type(message):
                            raise TypeError(f"on_message must return a {type(message).__name__}, got {result!r}")
                        yield result

        return (apply,)


def context(function: Callable[..., str | Awaitable[str]]) -> Context:
    """Append the returned text to the system prompt of every provider call."""
    return Context(function)


def on_input(function: Callable[..., UserMessage | Awaitable[UserMessage]]) -> OnInput:
    """Rewrite, or reject by raising, each new user message once per run."""
    return OnInput(function)


def on_message(function: Callable[..., Message | Awaitable[Message]]) -> OnMessage:
    return OnMessage(function)


def flatten(capabilities: Sequence[Capability | Filter]) -> tuple[Filter, ...]:
    return tuple(f for c in capabilities for f in (c.filters() if isinstance(c, Capability) else (c,)))


async def _resolve[T](value: T | Awaitable[T]) -> T:
    return await value if isinstance(value, Awaitable) else value
