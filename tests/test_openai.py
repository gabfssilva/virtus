import io
import os
import wave
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from openai import omit
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaAudio,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel

from virtus import (
    AssistantMessage,
    Audio,
    AudioDelta,
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

    def it_sends_audio_as_input_audio_parts():
        history = (UserMessage("Transcribe this.", audio=(Audio(b"RIFF", "audio/wav"),)),)

        assert _messages("", history) == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this."},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                ],
            }
        ]

    def it_sends_a_spoken_answer_back_as_its_transcript():
        history = (AssistantMessage("Hi there", audio=Audio(b"RIFF", "audio/wav")),)

        assert _messages("", history) == [{"role": "assistant", "content": "Hi there"}]

    def it_sends_tool_result_images_in_a_user_message_after_the_results():
        picture = Image(b"\x89PNG", "image/png")
        history = (
            AssistantMessage(tool_calls=(ToolCall("c1", "snapshot", "{}"), ToolCall("c2", "time", "{}"))),
            ToolResult("c1", "snapshot", "[image]", images=(picture,)),
            ToolResult("c2", "time", "noon"),
        )

        assert _messages("", history)[1:] == [
            {"role": "tool", "tool_call_id": "c1", "content": "[image]"},
            {"role": "tool", "tool_call_id": "c2", "content": "noon"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Media returned by the tool calls above."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}},
                ],
            },
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


    async def it_speaks_the_answer_when_audio_is_requested():
        chunks = [
            chunk(ChoiceDelta(audio=ChoiceDeltaAudio(id="a1", transcript="Hello "))),
            chunk(ChoiceDelta(audio=ChoiceDeltaAudio(data="UklG", transcript="there"))),
            chunk(ChoiceDelta(audio=ChoiceDeltaAudio(data="Rg=="))),
            chunk(ChoiceDelta(audio=ChoiceDeltaAudio(expires_at=1))),
        ]
        provider = OpenAI(voice="echo")
        client = Replaying(chunks, chunks)
        provider.__dict__["_client"] = client

        @agent(provider=provider)
        async def speak(text: str) -> Audio: ...

        @agent(provider=provider)
        async def watch(text: str) -> AsyncIterator[Event]: ...

        clip = await speak("hi")
        with wave.open(io.BytesIO(clip.data)) as file:
            assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 24_000)
            assert file.readframes(2) == b"RIFF"
        assert clip.media_type == "audio/wav"
        assert client.calls[0]["audio"] == {"voice": "echo", "format": "pcm16"}
        assert client.calls[0]["extra_body"] == {"modalities": ["text", "audio"]}
        assert [e async for e in watch("hi")] == [
            TextDelta("Hello "),
            TextDelta("there"),
            AudioDelta(b"RIF"),
            AudioDelta(b"F"),
            AssistantMessage("Hello there", audio=clip),
        ]

    async def it_reads_the_images_a_server_streams_in_the_delta():
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}}
        delta = ChoiceDelta.model_validate({"content": "Here", "images": [part]})
        provider = OpenAI()
        client = Replaying([chunk(delta)])
        provider.__dict__["_client"] = client

        @agent(provider=provider)
        async def draw(subject: str) -> Image: ...

        assert await draw("a cat") == Image(b"\x89PNG", "image/png")
        assert client.calls[0]["extra_body"] == {"modalities": ["text", "image"]}
        assert client.calls[0]["audio"] is omit


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
