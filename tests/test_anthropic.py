from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from anthropic.types import (
    InputJSONDelta,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    RawMessageStreamEvent,
    TextBlock,
    ToolUseBlock,
)
from anthropic.types import TextDelta as AnthropicTextDelta
from anthropic.types.raw_message_delta_event import Delta
from pydantic import BaseModel

from virtus import AssistantMessage, Event, TextDelta, ToolCall, ToolCallDelta, ToolResult, UserMessage, agent, tool
from virtus.providers import Anthropic
from virtus.providers.anthropic import messages


def describe_messages():
    def it_converts_the_conversation_and_groups_parallel_tool_results():
        history = (
            UserMessage("Weather?"),
            AssistantMessage("Checking.", (ToolCall("c1", "weather", '{"city": "Paris"}'), ToolCall("c2", "time", ""))),
            ToolResult("c1", "weather", "Sunny"),
            ToolResult("c2", "time", "boom", is_error=True),
            AssistantMessage("It's sunny"),
        )

        assert messages(history) == [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "c1", "name": "weather", "input": {"city": "Paris"}},
                    {"type": "tool_use", "id": "c2", "name": "time", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "Sunny", "is_error": False},
                    {"type": "tool_result", "tool_use_id": "c2", "content": "boom", "is_error": True},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "It's sunny"}]},
        ]


def describe_streaming():
    async def it_emits_deltas_and_the_assembled_message():
        first_turn: list[RawMessageStreamEvent] = [
            start(0, TextBlock(type="text", text="")),
            delta(0, AnthropicTextDelta(type="text_delta", text="Let me check.")),
            stop(0),
            start(1, ToolUseBlock(type="tool_use", id="c1", name="weather", input={})),
            delta(1, InputJSONDelta(type="input_json_delta", partial_json='{"city": ')),
            start(2, ToolUseBlock(type="tool_use", id="c2", name="time", input={})),
            delta(1, InputJSONDelta(type="input_json_delta", partial_json='"Paris"}')),
            stop(1),
            stop(2),
            RawMessageDeltaEvent(
                type="message_delta", delta=Delta(stop_reason="tool_use"), usage=MessageDeltaUsage(output_tokens=1)
            ),
            RawMessageStopEvent(type="message_stop"),
        ]
        second_turn: list[RawMessageStreamEvent] = [
            start(0, TextBlock(type="text", text="")),
            delta(0, AnthropicTextDelta(type="text_delta", text="done")),
            stop(0),
            RawMessageStopEvent(type="message_stop"),
        ]
        provider = Anthropic(effort="low")
        client = Replaying(first_turn, second_turn)
        provider.__dict__["_client"] = client

        @tool
        def weather(city: str) -> str:
            return f"sunny in {city}"

        @agent(provider=provider, system="Be brief.", capabilities=[weather])
        async def bot(message: str) -> AsyncIterator[Event]: ...

        events = [e async for e in bot("weather?")]

        assert events == [
            TextDelta("Let me check."),
            ToolCallDelta("c1", "weather", '{"city": '),
            ToolCallDelta("c1", "weather", '"Paris"}'),
            AssistantMessage(
                "Let me check.", (ToolCall("c1", "weather", '{"city": "Paris"}'), ToolCall("c2", "time", "{}"))
            ),
            ToolResult("c1", "weather", "sunny in Paris"),
            ToolResult("c2", "time", "Unknown tool: time", is_error=True),
            TextDelta("done"),
            AssistantMessage("done"),
        ]
        first_request = client.calls[0]
        assert first_request["system"] == "Be brief."
        assert first_request["output_config"] == {"effort": "low"}
        assert first_request["tools"] == [{"name": "weather", "description": "", "input_schema": weather.schema}]

    async def it_raises_on_refusal():
        provider = Anthropic()
        provider.__dict__["_client"] = Replaying(
            [
                RawMessageDeltaEvent(
                    type="message_delta", delta=Delta(stop_reason="refusal"), usage=MessageDeltaUsage(output_tokens=0)
                )
            ]
        )

        @agent(provider=provider)
        async def bot(message: str) -> str: ...

        with pytest.raises(RuntimeError, match="refused"):
            await bot("?")


def start(index: int, block: TextBlock | ToolUseBlock) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)


def delta(index: int, content: AnthropicTextDelta | InputJSONDelta) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(type="content_block_delta", index=index, delta=content)


def stop(index: int) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent(type="content_block_stop", index=index)


class Replaying:
    """Stands in for `AsyncAnthropic`: each `messages.create` records its arguments and streams the next turn."""

    def __init__(self, *turns: list[RawMessageStreamEvent]) -> None:
        self.messages = SimpleNamespace(create=self.create)
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    async def create(self, **arguments: object) -> AsyncIterator[RawMessageStreamEvent]:
        async def stream(events: list[RawMessageStreamEvent]) -> AsyncIterator[RawMessageStreamEvent]:
            for event in events:
                yield event

        self.calls.append(arguments)
        return stream(self.turns.pop(0))


class Forecast(BaseModel):
    city: str
    sunny: bool


@pytest.mark.integration
def describe_anthropic():
    async def it_calls_tools_and_returns_structured_output():
        @tool
        def weather(city: str) -> str:
            """Current weather for a city."""
            return f"Sunny and 25°C in {city}"

        @agent(provider=Anthropic(effort="low"), capabilities=[weather])
        async def forecast(city: str) -> Forecast:
            """Use the weather tool to tell whether it is sunny in {city}."""
            ...

        assert await forecast("Paris") == Forecast(city="Paris", sunny=True)
