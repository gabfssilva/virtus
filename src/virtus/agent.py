import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, overload
from uuid import uuid4

from pydantic import BaseModel

from virtus.capability import flatten
from virtus.dispatch import Target, dispatching
from virtus.journal import Journal, MemoryJournal
from virtus.message import Audio, Image, UserMessage, placeholder
from virtus.middleware import Capability, Middleware
from virtus.output import Output, output
from virtus.provider import Provider
from virtus.thread import Conversation, Thread
from virtus.tool import parameters_model


class Unset:
    """Marks a `replace` setting to keep."""


UNSET: Final = Unset()


class Agent[**P, R](Capability):
    """An LLM call loop described by a template function.

    The template's docstring, formatted with the call arguments, is the user message. Its return
    annotation decides what a call returns; see `virtus.output`. `R` is that return type, so an
    agent declared `-> Person` has `R = Awaitable[Person]` and one declared `-> AsyncIterator[Person]`
    has `R = AsyncIterator[Person]`.

    Used as a capability, the agent becomes a target of the caller's `dispatch` tool (`virtus.dispatch`):
    the calling model addresses it by name for a one-off request, or as `name:thread` to keep a
    conversation going, and passes the template's arguments, which format the message exactly as a
    direct call does. Those threads belong to the caller's thread, so separate conversations never
    share them.
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
        self.journal: Journal = journal or MemoryJournal()
        """Where every thread of this agent keeps its history; one-off calls are never written to it."""
        self.output: Final[Output[R]] = output
        self.capabilities: Final[tuple[Capability | Middleware, ...]] = tuple(capabilities)
        self.chain: tuple[Middleware, ...] = flatten(capabilities)
        """Middleware applied around each step of a run, outermost first."""
        self.parameters: type[BaseModel] = parameters_model(template)
        self._template = template
        self._signature = signature
        self._prompt = inspect.getdoc(template) or "\n\n".join(f"{{{p}}}" for p in signature.parameters)

        if not self._prompt:
            raise TypeError(f"{self.name}: an agent needs parameters or a docstring")

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
        journal: Journal | Unset = UNSET,
    ) -> "Agent[P, R]":
        """A copy with these settings changed; same template and output mode, and the same journal unless
        one is given, so both see the same threads.

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
        """A handle to the conversation with this id in the journal, new or already under way.

        Without an id, start a new one.
        """
        return Thread(self, (id or uuid4().hex,), self.journal)

    def input(self, *args: object, **kwargs: object) -> UserMessage:
        """The user message a call with these arguments sends.

        `Image` and `Audio` arguments, alone or in a sequence, are attached to the message; in the
        prompt they read as `[image]` and `[audio]`.
        """
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        values = list(bound.arguments.values())
        images = tuple(image for value in values for image in Image.found_in(value))
        audio = tuple(clip for value in values for clip in Audio.found_in(value))
        text = {name: placeholder(value) or value for name, value in bound.arguments.items()}
        return UserMessage(self._prompt.format(**text), images, audio)

    def middleware(self) -> tuple[Middleware, ...]:
        def conversation(parent: Conversation, thread: str | None) -> Conversation:
            return parent.child((self.name, thread), self) if thread else Thread(self)

        return (dispatching(Target(self, conversation)),)


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
