"""How an agent's return annotation decides what a call returns.

`-> T`                     awaitable: the final answer parsed as `T`
`-> AsyncIterator[str]`    the answer's text as it is generated
`-> AsyncIterator[Event]`  the raw events filters see; a union of event types keeps only those
`-> AsyncIterator[T]`      the answer is a `list[T]`; each item is emitted as soon as it is complete
`-> Snapshot[T]`           the answer is a `T`, emitted again with more of it filled in as it grows
"""

import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from operator import or_
from types import GenericAlias, NoneType, UnionType
from typing import Protocol, TypeAliasType, Union, cast, get_args, get_origin, get_type_hints, runtime_checkable

from pydantic import BaseModel, TypeAdapter, ValidationError, create_model
from pydantic_core import from_json
from typing_extensions import TypeForm

from virtus.message import AssistantMessage, Event, Message, TextDelta, ToolCallDelta

type Snapshot[T] = AsyncIterator[T]
"""Return annotation for a `T` streamed as successive, increasingly complete versions of itself."""


@runtime_checkable
class Output[R](Protocol):
    @property
    def schema(self) -> Mapping[str, object] | None:
        """JSON schema requested from the provider; `None` for plain text.

        Implementations are frozen dataclasses whose `schema` field needs a default: the class-level
        default is what shadows this property, otherwise their `__init__` would try to set it.
        """
        ...

    def consume(self, events: AsyncIterator[Event]) -> R:
        """What a call to the agent returns, built from the run's events."""
        ...

    async def final(self, events: AsyncIterator[Event]) -> object:
        """The complete answer, for callers that need a single value such as delegation."""
        ...


def output(template: Callable[..., object]) -> Output[object]:
    """Pick the output mode from the template's return annotation."""
    hint = get_type_hints(template).get("return", str)
    origin, args = get_origin(hint), get_args(hint)

    if origin is not Snapshot and origin is not AsyncIterator:
        return Value(TypeAdapter(hint), None if hint is str else _schema(hint))

    (inner,) = args
    if origin is Snapshot:
        if inner is str:
            return Snapshots(TypeAdapter(str), TypeAdapter(str), None)
        return Snapshots(TypeAdapter(inner), TypeAdapter(_partial(inner)), _schema(inner))
    if inner is str:
        return Text()
    if all(member in _EVENTS for member in _members(inner)):
        return Events(tuple(member for member in _members(inner) if isinstance(member, type)))
    return Items(TypeAdapter(inner), _schema(_form(GenericAlias(list, (inner,)))))


@dataclass(frozen=True)
class Value[Out](Output[Awaitable[Out]]):
    adapter: TypeAdapter[Out]
    schema: Mapping[str, object] | None = None

    def consume(self, events: AsyncIterator[Event]) -> Awaitable[Out]:
        return self.final(events)

    async def final(self, events: AsyncIterator[Event]) -> Out:
        content = await _answer(events)
        return self.adapter.validate_python(content if self.schema is None else from_json(content)["value"])


@dataclass(frozen=True)
class Text(Output[AsyncIterator[str]]):
    schema: Mapping[str, object] | None = None

    async def consume(self, events: AsyncIterator[Event]) -> AsyncIterator[str]:
        async for event in events:
            if isinstance(event, TextDelta):
                yield event.content

    async def final(self, events: AsyncIterator[Event]) -> str:
        return await _answer(events)


@dataclass(frozen=True)
class Events(Output[AsyncIterator[Event]]):
    kinds: tuple[type, ...]
    schema: Mapping[str, object] | None = None

    async def consume(self, events: AsyncIterator[Event]) -> AsyncIterator[Event]:
        async for event in events:
            if isinstance(event, self.kinds):
                yield event

    async def final(self, events: AsyncIterator[Event]) -> str:
        return await _answer(events)


@dataclass(frozen=True)
class Items[T](Output[AsyncIterator[T]]):
    item: TypeAdapter[T]
    schema: Mapping[str, object] | None = None

    async def consume(self, events: AsyncIterator[Event]) -> AsyncIterator[T]:
        """Emit every item but the last of each partial parse, since the last one may still be open."""
        emitted = 0

        async for text in _answers(events):
            match text:
                case AssistantMessage(content=content):
                    for item in self._items(content)[emitted:]:
                        yield item
                    return
                case str():
                    closed = _partial_value(text)
                    if isinstance(closed, list):
                        for raw in closed[emitted:-1]:
                            yield self.item.validate_python(raw)
                            emitted += 1

    async def final(self, events: AsyncIterator[Event]) -> list[T]:
        return self._items(await _answer(events))

    def _items(self, content: str) -> list[T]:
        return [self.item.validate_python(raw) for raw in from_json(content)["value"]]


@dataclass(frozen=True)
class Snapshots[T](Output[AsyncIterator[T]]):
    """Snapshots that do not yet satisfy `T` validate against a copy of it whose fields are all optional.

    In those, missing fields are `None` and the last string may be cut short. As soon as the JSON
    so far satisfies `T`, snapshots are plain `T` values; the last one always is.
    """

    strict: TypeAdapter[T]
    partial: TypeAdapter[T]
    schema: Mapping[str, object] | None = None

    async def consume(self, events: AsyncIterator[Event]) -> AsyncIterator[T]:
        previous: T | None = None

        async for text in _answers(events):
            match text:
                case AssistantMessage(content=content):
                    final = self._final(content)
                    if final != previous:
                        yield final
                    return
                case str():
                    snapshot = self.partial.validate_python(text) if self.schema is None else self._snapshot(text)
                    if snapshot is not None and snapshot != previous:
                        previous = snapshot
                        yield snapshot

    async def final(self, events: AsyncIterator[Event]) -> T:
        return self._final(await _answer(events))

    def _final(self, content: str) -> T:
        return self.strict.validate_python(content if self.schema is None else from_json(content)["value"])

    def _snapshot(self, text: str) -> T | None:
        value = _partial_value(text)
        if value is None:
            return None
        for adapter in (self.strict, self.partial):
            try:
                return adapter.validate_python(value)
            except ValidationError:
                continue
        return None


async def _answer(events: AsyncIterator[Event]) -> str:
    """Drain the run and return the content of its final assistant message."""
    last: Message | None = None

    async for event in events:
        if not isinstance(event, (TextDelta, ToolCallDelta)):
            last = event

    match last:
        case AssistantMessage(content=content, tool_calls=()):
            return content
        case _:
            raise RuntimeError(f"the run ended without a final answer, last message: {last!r}")


async def _answers(events: AsyncIterator[Event]) -> AsyncIterator[str | AssistantMessage]:
    """The text of the assistant message in progress after each delta, then the final message itself."""
    text: list[str] = []
    last: Message | None = None

    async for event in events:
        match event:
            case TextDelta(content=content):
                text.append(content)
                yield "".join(text)
            case AssistantMessage(tool_calls=()):
                yield event
                return
            case ToolCallDelta():
                pass
            case _:
                text.clear()
                last = event

    raise RuntimeError(f"the run ended without a final answer, last message: {last!r}")


_UNTERMINATED_NUMBER = re.compile(r"[-+]?\d[-+.eE\d]*$")


def _partial_value(text: str) -> object | None:
    """Parse the JSON generated so far and return its `value`, or `None` while nothing parses yet.

    A number at the very end is dropped: `25` cut after the `2` would otherwise read as a valid `2`.
    """
    try:
        data = from_json(_UNTERMINATED_NUMBER.sub("", text), allow_partial="trailing-strings")
    except ValueError:
        return None
    return data.get("value") if isinstance(data, dict) else None


def _schema(annotation: TypeForm[object]) -> Mapping[str, object]:
    return _closed(create_model("output", value=(annotation, ...)).model_json_schema())


def _closed(schema: Mapping[str, object]) -> dict[str, object]:
    """Every object gets `additionalProperties: false`; strict decoders such as OpenAI's reject open ones."""

    def walk(value: object) -> object:
        match value:
            case Mapping():
                return _closed(value)
            case list():
                return [walk(item) for item in value]
            case _:
                return value

    closed = {key: walk(value) for key, value in schema.items()}
    return {**closed, "additionalProperties": False} if closed.get("type") == "object" else closed


def _partial(annotation: TypeForm[object]) -> TypeForm[object]:
    """The same type with every model field optional, recursively, so incomplete JSON validates."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields = {
            name: or_(_partial(field.annotation or object), NoneType) for name, field in annotation.model_fields.items()
        }
        namespace = {"__annotations__": fields, **dict.fromkeys(fields)}
        return _form(type(f"Partial{annotation.__name__}", (annotation,), namespace))

    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Union or origin is UnionType:
        return _form(reduce(or_, (_partial(member) for member in args)))
    if isinstance(origin, type) and args:
        return _form(GenericAlias(origin, tuple(_partial(member) for member in args)))
    return annotation


def _form(value: object) -> TypeForm[object]:
    """A type assembled at runtime is a type form to pydantic but, per PEP 747, not to the checker."""
    return cast(TypeForm[object], value)


def _members(annotation: object) -> Iterator[object]:
    if isinstance(annotation, TypeAliasType):
        yield from _members(annotation.__value__)
    elif get_origin(annotation) is Union or get_origin(annotation) is UnionType:
        for member in get_args(annotation):
            yield from _members(member)
    else:
        yield annotation


_EVENTS: Sequence[object] = tuple(_members(Event))
