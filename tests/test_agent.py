import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Audio,
    Event,
    Image,
    MemoryJournal,
    Next,
    Request,
    TextDelta,
    ToolResult,
    UserMessage,
    agent,
    tool,
)


class City(BaseModel):
    name: str
    country: str


def describe_agent():
    async def it_sends_the_docstring_formatted_with_the_arguments():
        provider = Scripted(AssistantMessage("Sunny"))

        @agent(provider=provider, system="Be brief.")
        async def weather(city: str, unit: str = "celsius") -> str:
            """Weather in {city}, in {unit}?"""
            ...

        assert await weather("Paris") == "Sunny"
        assert provider.requests[0].system == "Be brief."
        assert provider.requests[0].messages == (UserMessage("Weather in Paris, in celsius?"),)

    async def it_joins_the_arguments_when_there_is_no_docstring():
        provider = Scripted(AssistantMessage("ok"))

        @agent(provider=provider)
        async def echo(first: str, second: str) -> str: ...

        await echo("a", "b")

        assert provider.requests[0].messages == (UserMessage("a\n\nb"),)

    def when_the_output_is_not_text():
        async def it_requests_and_parses_a_model():
            provider = Scripted(AssistantMessage(json.dumps({"value": {"name": "Paris", "country": "France"}})))

            @agent(provider=provider)
            async def capital(country: str) -> City:
                """Capital of {country}"""
                ...

            assert await capital("France") == City(name="Paris", country="France")
            assert provider.requests[0].output is not None

        async def it_wraps_non_object_types():
            provider = Scripted(AssistantMessage('{"value": 42}'))

            @agent(provider=provider)
            async def answer(question: str) -> int:
                """{question}"""
                ...

            assert await answer("?") == 42

    async def it_fails_when_the_run_ends_without_a_final_answer():
        provider = Scripted(calls(("missing", {})), AssistantMessage("never reached"))

        async def cut(request: Request, next: Next) -> AsyncIterator[Event]:
            async for event in next(request):
                if not isinstance(event, ToolResult):
                    yield event

        @agent(provider=provider, capabilities=[cut])
        async def chat(prompt: str) -> str: ...

        with pytest.raises(RuntimeError, match="without a final answer"):
            await chat("hi")


def describe_stream():
    async def it_yields_deltas_messages_and_tool_results_in_order():
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        provider = Scripted(calls(("add", {"a": 1, "b": 2})), AssistantMessage("It is 3"))

        @agent(provider=provider, capabilities=[add])
        async def math(question: str) -> AsyncIterator[Event]: ...

        events = [event async for event in math("1 + 2?")]

        assert events == [
            calls(("add", {"a": 1, "b": 2})),
            ToolResult("call-0", "add", "3"),
            TextDelta("It "),
            TextDelta("is "),
            TextDelta("3"),
            AssistantMessage("It is 3"),
        ]


def describe_dispatch():
    async def it_exposes_an_agent_through_the_dispatch_tool_with_its_description():
        research_provider = Scripted(AssistantMessage("Bees dance"))

        @agent(provider=research_provider, description="Researches a topic")
        async def researcher(topic: str) -> str:
            """Research {topic}"""
            ...

        delivery = calls(("dispatch", {"to": ["researcher"], "arguments": {"topic": "bees"}}))
        lead_provider = Scripted(delivery, AssistantMessage("Done"))

        @agent(provider=lead_provider, capabilities=[researcher])
        async def lead(task: str) -> str: ...

        assert await lead("write about bees") == "Done"

        [dispatch] = lead_provider.requests[0].tools
        assert dispatch.name == "dispatch"
        assert list(dispatch.parameters.model_fields) == ["to", "arguments"]
        assert dispatch.schema["properties"] == {
            "to": {
                "description": "Addresses: `agent` or `agent:conversation`.",
                "items": {"type": "string"},
                "minItems": 1,
                "title": "To",
                "type": "array",
            },
            "arguments": {
                "additionalProperties": True,
                "description": "The arguments of the addressed agent.",
                "title": "researcher",
                "type": "object",
                "properties": {"topic": {"title": "Topic", "type": "string"}},
                "required": ["topic"],
            },
        }
        assert lead_provider.requests[0].system.endswith("- researcher: Researches a topic")
        assert research_provider.requests[0].messages == (UserMessage("Research bees"),)
        assert lead_provider.requests[1].messages[-1] == ToolResult("call-0", "dispatch", '{"researcher":"Bees dance"}')


def describe_images():
    def it_attaches_audio_arguments_and_names_them_in_the_prompt():
        clip = Audio(b"RIFF", "audio/wav")

        @agent(provider=Scripted())
        async def transcribe(recording: Audio, language: str) -> str:
            """Transcribe {recording} in {language}."""
            ...

        assert transcribe.input(clip, "pt") == UserMessage("Transcribe [audio] in pt.", audio=(clip,))

    def it_attaches_image_arguments_and_names_them_in_the_prompt():
        picture = Image(b"\x89PNG", "image/png")
        provider = Scripted(AssistantMessage("a cat"))

        @agent(provider=provider)
        async def describe(photo: Image, question: str, more: list[Image] = []) -> str:  # noqa: B006
            """{question} ({photo})"""
            ...

        message = describe.input(picture, "What is this?", [picture, picture])

        assert message == UserMessage("What is this? ([image])", (picture, picture, picture))

    def it_reads_files_and_rejects_unknown_types(tmp_path: Path):
        file = tmp_path / "pixel.png"
        file.write_bytes(b"\x89PNG")

        assert Image.read(file) == Image(b"\x89PNG", "image/png")
        with pytest.raises(ValueError, match="unsupported image type"):
            Image.read(tmp_path / "notes.txt")


def describe_replace():
    class Pet(BaseModel):
        name: str

    async def it_copies_the_agent_with_some_settings_changed():
        first = Scripted(AssistantMessage('{"value": {"name": "Rex"}}'))
        second = Scripted(AssistantMessage('{"value": {"name": "Max"}}'))

        @tool
        def bark() -> str:
            return "woof"

        @agent(provider=first, system="Be brief.")
        async def pet(kind: str) -> Pet:
            """Name a {kind}."""
            ...

        other = pet.replace(provider=second, capabilities=[bark])

        assert await pet("dog") == Pet(name="Rex")
        assert await other("dog") == Pet(name="Max")
        assert first.requests[0].system == "Be brief."
        assert second.requests[0].system == "Be brief."
        assert [t.name for t in second.requests[0].tools] == ["bark"]
        assert pet.chain == ()

    async def it_shares_the_journal_unless_one_is_given():
        @agent(provider=Scripted(AssistantMessage("hi")))
        async def chat(message: str) -> str: ...

        await chat.thread("u1")("hello")
        same = chat.replace(system="Be brief.")
        apart = chat.replace(journal=MemoryJournal())

        assert same.thread("u1").agent is same
        assert same.thread("u1").history == chat.thread("u1").history
        assert apart.thread("u1").history == ()

    async def it_extends_the_capabilities_in_order():
        provider = Scripted(AssistantMessage("ok"))

        @tool
        def first() -> str:
            return "1"

        @tool
        def second() -> str:
            return "2"

        @agent(provider=Scripted(), capabilities=[first])
        async def chat(message: str) -> str: ...

        other = chat.extend(second).replace(provider=provider)

        assert await other("hi") == "ok"
        assert [t.name for t in provider.requests[0].tools] == ["first", "second"]
        assert chat.capabilities == (first,)
