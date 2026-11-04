import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Delta,
    Event,
    Provider,
    Request,
    Snapshot,
    TextDelta,
    ToolResult,
    UserMessage,
    agent,
    tool,
)


class Person(BaseModel):
    name: str
    age: int


class Paper(BaseModel):
    title: str
    authors: list[Person]


class Chunked(Provider):
    """Streams the reply in the given pieces and waits for a go-ahead before each one after the first."""

    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.sent = 0
        self.proceed = asyncio.Event()

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        for chunk in self.chunks:
            if self.sent:
                await self.proceed.wait()
                self.proceed.clear()
            self.sent += 1
            yield TextDelta(chunk)
        yield AssistantMessage("".join(self.chunks))


def _answer(value: object) -> str:
    return json.dumps({"value": value})


def describe_text():
    async def it_yields_the_text_as_it_is_generated():
        @agent(provider=Scripted(AssistantMessage("one two three")))
        async def chat(message: str) -> AsyncIterator[str]: ...

        assert [t async for t in chat("hi")] == ["one ", "two ", "three"]


def describe_events():
    async def it_keeps_only_the_requested_kinds():
        @tool
        def ping() -> str:
            return "pong"

        provider = Scripted(calls(("ping", {})), AssistantMessage("done"))

        @agent(provider=provider, capabilities=[ping])
        async def chat(message: str) -> AsyncIterator[ToolResult | UserMessage]: ...

        assert [e async for e in chat("hi")] == [ToolResult("call-0", "ping", "pong")]

    async def it_yields_everything_for_event():
        @agent(provider=Scripted(AssistantMessage("a b")))
        async def chat(message: str) -> AsyncIterator[Event]: ...

        assert [e async for e in chat("hi")] == [TextDelta("a "), TextDelta("b"), AssistantMessage("a b")]


def describe_items():
    async def it_requests_a_list_and_yields_each_item():
        people = [{"name": "Ana", "age": 30}, {"name": "Bia", "age": 25}]
        provider = Scripted(AssistantMessage(_answer(people)))

        @agent(provider=provider)
        async def listing(topic: str) -> AsyncIterator[Person]: ...

        assert [p async for p in listing("x")] == [Person(name="Ana", age=30), Person(name="Bia", age=25)]
        assert provider.requests[0].output == {
            "properties": {
                "value": {"items": {"$ref": "#/$defs/Person"}, "title": "Value", "type": "array"},
            },
            "required": ["value"],
            "title": "output",
            "type": "object",
            "additionalProperties": False,
            "$defs": {
                "Person": {
                    "properties": {
                        "name": {"title": "Name", "type": "string"},
                        "age": {"title": "Age", "type": "integer"},
                    },
                    "required": ["name", "age"],
                    "title": "Person",
                    "type": "object",
                    "additionalProperties": False,
                }
            },
        }

    async def it_yields_an_item_as_soon_as_the_next_one_starts():
        provider = Chunked('{"value": [{"name": "Ana", "age": 30}, {"na', 'me": "Bia", "age": 25}]}')

        @agent(provider=provider)
        async def listing(topic: str) -> AsyncIterator[Person]: ...

        items = listing("x")

        assert await anext(items) == Person(name="Ana", age=30)
        assert provider.sent == 1

        provider.proceed.set()
        assert await anext(items) == Person(name="Bia", age=25)

    async def it_is_a_list_when_delegated_to():
        research = Scripted(AssistantMessage(_answer([{"name": "Ana", "age": 30}])))

        @agent(provider=research)
        async def listing(topic: str) -> AsyncIterator[Person]: ...

        lead_provider = Scripted(calls(("listing", {"topic": "x"})), AssistantMessage("done"))

        @agent(provider=lead_provider, capabilities=[listing])
        async def lead(message: str) -> str: ...

        await lead("go")

        assert lead_provider.requests[1].messages[-1] == ToolResult("call-0", "listing", '[{"name":"Ana","age":30}]')


def describe_snapshot():
    async def it_fills_the_object_in_as_the_json_arrives():
        provider = Chunked('{"value": {"name": "An', 'a", "age": 3', "0}}")

        @agent(provider=provider)
        async def bio(name: str) -> Snapshot[Person]: ...

        snapshots = bio("x")

        first = await anext(snapshots)
        assert isinstance(first, Person)
        assert (first.name, first.age) == ("An", None)

        provider.proceed.set()
        second = await anext(snapshots)
        assert (second.name, second.age) == ("Ana", None)

        provider.proceed.set()
        third = await anext(snapshots)
        assert third == Person(name="Ana", age=30)
        assert type(third) is Person
        with pytest.raises(StopAsyncIteration):
            await anext(snapshots)

    async def it_fills_nested_models_and_lists():
        paper = {"title": "Bees", "authors": [{"name": "Ana", "age": 30}, {"name": "Bia", "age": 25}]}
        provider = Scripted(AssistantMessage(_answer(paper)))

        @agent(provider=provider)
        async def summarize(topic: str) -> Snapshot[Paper]: ...

        snapshots = [s async for s in summarize("x")]

        assert snapshots[-1] == Paper(title="Bees", authors=[Person(name="Ana", age=30), Person(name="Bia", age=25)])
        assert all(isinstance(s, Paper) for s in snapshots)
        assert any(s.title == "Bees" and s.authors is None for s in snapshots)
        assert any(s.authors is not None and len(s.authors) == 2 and s.authors[1].age is None for s in snapshots)

    async def it_requests_the_schema_of_the_type_itself():
        provider = Scripted(AssistantMessage(_answer({"name": "Ana", "age": 30})))

        @agent(provider=provider)
        async def bio(name: str) -> Snapshot[Person]: ...

        [s async for s in bio("x")]

        schema = provider.requests[0].output
        assert schema is not None
        assert schema["properties"] == {"value": {"$ref": "#/$defs/Person"}}

    async def it_closes_every_object_for_strict_decoders():
        provider = Scripted(AssistantMessage(_answer({"title": "Bees", "authors": []})))

        @agent(provider=provider)
        async def summarize(topic: str) -> Snapshot[Paper]: ...

        [s async for s in summarize("x")]

        schema = provider.requests[0].output
        assert schema is not None
        assert schema["additionalProperties"] is False
        assert json.dumps(schema).count('"additionalProperties": false') == 3  # output, Paper, Person

    async def it_accumulates_text_for_str():
        @agent(provider=Scripted(AssistantMessage("one two")))
        async def chat(message: str) -> Snapshot[str]: ...

        assert [s async for s in chat("hi")] == ["one ", "one two"]
