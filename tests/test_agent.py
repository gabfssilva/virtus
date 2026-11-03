import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, Event, Image, Next, Request, TextDelta, ToolResult, UserMessage, agent, tool


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


def describe_delegation():
    async def it_exposes_an_agent_as_a_tool_with_its_parameters():
        research_provider = Scripted(AssistantMessage("Bees dance"))

        @agent(provider=research_provider, description="Researches a topic")
        async def researcher(topic: str) -> str:
            """Research {topic}"""
            ...

        lead_provider = Scripted(calls(("researcher", {"topic": "bees"})), AssistantMessage("Done"))

        @agent(provider=lead_provider, capabilities=[researcher])
        async def lead(task: str) -> str: ...

        assert await lead("write about bees") == "Done"

        [delegate] = lead_provider.requests[0].tools
        assert (delegate.name, delegate.description) == ("researcher", "Researches a topic")
        assert research_provider.requests[0].messages == (UserMessage("Research bees"),)
        assert lead_provider.requests[1].messages[-1] == ToolResult("call-0", "researcher", "Bees dance")


def describe_images():
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
