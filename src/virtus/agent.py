import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, overload
from uuid import uuid4

from pydantic import BaseModel, Field, create_model

from virtus.capability import flatten
from virtus.journal import Journal
from virtus.message import Event, Image, UserMessage
from virtus.middleware import Capability, Middleware, Next
from virtus.output import Output, output
from virtus.provider import Provider
from virtus.request import Request
from virtus.thread import Thread
from virtus.tool import Tool, parameters_model


class Unset:
    """Marks a `replace` setting to keep; `None` is a real value for some of them."""


UNSET: Final = Unset()


class Agent[**P, R](Capability):
    """An LLM call loop described by a template function.

    The template's docstring, formatted with the call arguments, is the user message. Its return
    annotation decides what a call returns; see `virtus.output`. `R` is that return type, so an
    agent declared `-> Person` has `R = Awaitable[Person]` and one declared `-> AsyncIterator[Person]`
    has `R = AsyncIterator[Person]`.

    Used as a capability, the agent becomes a tool other agents delegate to. The tool takes an
    optional `thread` name chosen by the calling model: reusing a name continues that conversation
    with this agent, omitting it makes a one-off request. These threads belong to the caller's
    thread, so separate conversations never share them.
    """

    def __init__(
        self,
        template: Callable[P, object],
        *,
        output: Output[R],
        provider: Provider,
        name: str | None = None,
        description: str | None = None,
        system: str = "",
        capabilities: Sequence[Capability | Middleware] = (),
        journal: Journal | None = None,
    ) -> None:
        signature = inspect.signature(template)

        self.name = name or template.__name__
        self.description = description or inspect.getdoc(template) or ""
        self.provider = provider
        self.system = system
        self.journal = journal
        """Where `thread(id)` keeps conversations across processes; one-off calls are never written."""
        self.output: Final[Output[R]] = output
        self.capabilities: Final[tuple[Capability | Middleware, ...]] = tuple(capabilities)
        self.chain: tuple[Middleware, ...] = flatten(capabilities)
        """Middleware applied around each step of a run, outermost first."""
        self.parameters: type[BaseModel] = parameters_model(template)
        self._template = template
        self._signature = signature
        self._prompt = inspect.getdoc(template) or "\n\n".join(f"{{{p}}}" for p in signature.parameters)
        self._threads: Mapping[str, Thread[P, R]] = {}

        if not self._prompt:
            raise TypeError(f"{self.name}: an agent needs parameters or a docstring")
        if "thread" in self.parameters.model_fields:
            raise TypeError(f"{self.name}: `thread` is reserved for delegation and can't be a parameter")

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return Thread(self)(*args, **kwargs)

    def replace(
        self,
        *,
        provider: Provider | Unset = UNSET,
        name: str | Unset = UNSET,
        description: str | Unset = UNSET,
        system: str | Unset = UNSET,
        capabilities: Sequence[Capability | Middleware] | Unset = UNSET,
        journal: Journal | Unset | None = UNSET,
    ) -> "Agent[P, R]":
        """A copy with these settings changed; same template and output mode, its own threads.

        `capabilities` replaces the whole list; `extend` appends to it.
        """

        def pick[T](value: T | Unset, current: T) -> T:
            return current if isinstance(value, Unset) else value

        return Agent(
            self._template,
            output=self.output,
            provider=pick(provider, self.provider),
            name=pick(name, self.name),
            description=pick(description, self.description),
            system=pick(system, self.system),
            capabilities=pick(capabilities, self.capabilities),
            journal=pick(journal, self.journal),
        )

    def extend(self, *capabilities: Capability | Middleware) -> "Agent[P, R]":
        """A copy with these capabilities appended after the current ones."""
        return self.replace(capabilities=(*self.capabilities, *capabilities))

    def thread(self, id: str | None = None) -> Thread[P, R]:
        """Return the thread with this id, from memory or the journal, creating it if needed.

        Without an id, create a new one.
        """
        key = id or uuid4().hex
        if found := self._threads.get(key):
            return found

        def detach() -> None:
            self._threads = {k: t for k, t in self._threads.items() if k != key}

        created = Thread(self, (key,), detach, journal=self.journal)
        self._threads = {**self._threads, key: created}
        return created

    def input(self, *args: object, **kwargs: object) -> UserMessage:
        """The user message a call with these arguments sends.

        `Image` arguments, alone or in a sequence, are attached to the message; in the prompt
        they read as `[image]`.
        """
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        images = tuple(image for value in bound.arguments.values() for image in Image.found_in(value))
        text = {name: "[image]" if Image.found_in(value) else value for name, value in bound.arguments.items()}
        return UserMessage(self._prompt.format(**text), images)

    def middleware(self) -> tuple[Middleware, ...]:
        parameters = create_model(
            self.name,
            __base__=self.parameters,
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
                target = parent.child((self.name, thread), self) if thread else Thread(self)
                return await target.send(self.input(**arguments))

            names = [name for agent, name in parent.children if agent == self.name]
            note = f"Open conversations with {self.name}: {', '.join(names)}" if names else ""
            system = "\n\n".join(part for part in (request.system, note) if part)
            tool = Tool(self.name, self.description, parameters, run)

            async for event in next(replace(request, system=system, tools=(*request.tools, tool))):
                yield event

        return (delegate,)


def agent(
    *,
    provider: Provider,
    name: str | None = None,
    description: str | None = None,
    system: str = "",
    capabilities: Sequence[Capability | Middleware] = (),
    journal: Journal | None = None,
) -> "Decorator":
    """Turn an async template function into an `Agent`; see `virtus.output` for the return types."""
    return Decorator(provider, name, description, system, capabilities, journal)


@dataclass(frozen=True)
class Decorator:
    provider: Provider
    name: str | None
    description: str | None
    system: str
    capabilities: Sequence[Capability | Middleware]
    journal: Journal | None

    @overload
    def __call__[**P, T](self, template: Callable[P, Awaitable[AsyncIterator[T]]]) -> Agent[P, AsyncIterator[T]]: ...
    @overload
    def __call__[**P, Out](self, template: Callable[P, Awaitable[Out]]) -> Agent[P, Awaitable[Out]]: ...
    def __call__[**P](self, template: Callable[P, Awaitable[object]]) -> Agent[P, object]:
        return Agent(
            template,
            output=output(template),
            provider=self.provider,
            name=self.name,
            description=self.description,
            system=self.system,
            capabilities=self.capabilities,
            journal=self.journal,
        )
