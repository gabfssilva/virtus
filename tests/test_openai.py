import os
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel

from virtus import (
    AssistantMessage,
    Event,
    Image,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    UserMessage,
    agent,
    tool,
)
from virtus.providers import OpenAI
from virtus.providers.openai import _messages


def describe_messages():
    def it_converts_the_conversation_to_chat_completions_format():
        history = (
            UserMessage("Weather?"),
            AssistantMessage(tool_calls=(ToolCall("c1", "weather", '{"city": "Paris"}'),)),
            ToolResult("c1", "weather", "Sunny"),
            AssistantMessage("It's sunny"),
        )

        assert _messages("Be brief.", history) == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
            {"role": "assistant", "content": "It's sunny"},
        ]

    def it_sends_images_as_data_urls_next_to_the_text():
        history = (UserMessage("What is this?", (Image(b"\x89PNG", "image/png"),)),)

        assert _messages("", history) == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}},
                ],
            }
        ]


class Forecast(BaseModel):
    city: str
    sunny: bool


@pytest.mark.integration
def describe_openai():
    async def it_calls_tools_and_returns_structured_output():
        assert os.environ.get("OPENAI_API_KEY"), "integration tests need OPENAI_API_KEY"

        @tool
        def weather(city: str) -> str:
            """Current weather for a city."""
            return f"Sunny and 25°C in {city}"

        @agent(provider=OpenAI(temperature=0), capabilities=[weather])
        async def forecast(city: str) -> Forecast:
            """Use the weather tool to tell whether it is sunny in {city}."""
            ...

        assert await forecast("Paris") == Forecast(city="Paris", sunny=True)


def describe_streaming():
    async def it_emits_a_tool_call_delta_per_fragment_and_the_assembled_call_at_the_end():
        def fragment(index: int, *, id: str | None = None, name: str | None = None, arguments: str | None = None):
            function = ChoiceDeltaToolCallFunction(name=name, arguments=arguments)
            return ChoiceDeltaToolCall(index=index, id=id, type="function", function=function)

        chunks = [
            chunk(ChoiceDelta(content="Let me check.")),
            chunk(ChoiceDelta(tool_calls=[fragment(0, id="c1", name="weather", arguments="")])),
            chunk(ChoiceDelta(tool_calls=[fragment(0, arguments='{"city": '), fragment(1, id="c2", name="time")])),
            chunk(ChoiceDelta(tool_calls=[fragment(0, arguments='"Paris"}'), fragment(1, arguments="{}")])),
        ]
        provider = OpenAI(max_tokens=300, reasoning_effort="low")
        client = Replaying(chunks, [chunk(ChoiceDelta(content="done"))])
        provider.__dict__["_client"] = client

        @agent(provider=provider)
        async def bot(message: str) -> AsyncIterator[Event]: ...

        events = [e async for e in bot("weather?")]
        first_step = events[: next(i for i, e in enumerate(events) if isinstance(e, AssistantMessage)) + 1]

        assert events[-1] == AssistantMessage("done")
        assert (client.calls[0]["max_completion_tokens"], client.calls[0]["reasoning_effort"]) == (300, "low")
        assert first_step == [
            TextDelta("Let me check."),
            ToolCallDelta("c1", "weather", ""),
            ToolCallDelta("c1", "weather", '{"city": '),
            ToolCallDelta("c2", "time", ""),
            ToolCallDelta("c1", "weather", '"Paris"}'),
            ToolCallDelta("c2", "time", "{}"),
            AssistantMessage(
                "Let me check.", (ToolCall("c1", "weather", '{"city": "Paris"}'), ToolCall("c2", "time", "{}"))
            ),
        ]


def chunk(delta: ChoiceDelta) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk", choices=[Choice(index=0, delta=delta)], created=0, model="test", object="chat.completion.chunk"
    )


class Replaying:
    """Stands in for `AsyncOpenAI`: each `chat.completions.create` streams the next list of chunks."""

    def __init__(self, *replies: list[ChatCompletionChunk]) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.replies = list(replies)
        self.calls: list[dict[str, object]] = []

    async def create(self, **arguments: object) -> AsyncIterator[ChatCompletionChunk]:
        async def stream(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
            for chunk in chunks:
                yield chunk

        self.calls.append(arguments)
        return stream(self.replies.pop(0))
