from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from openai import omit
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseRefusalDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_tool_search_call import ResponseToolSearchCall
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from virtus import (
    AssistantMessage,
    Audio,
    Event,
    File,
    Image,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    Usage,
    UserMessage,
    agent,
    lazy,
    tool,
)
from virtus.lazy import Search
from virtus.providers import Responses
from virtus.providers.responses import _input, _search, _tool


@tool
def weather(city: str) -> str:
    """Weather in a city."""
    return f"sunny in {city}"


def describe_input():
    def it_converts_the_conversation_to_input_items():
        history = (
            UserMessage("Weather?"),
            AssistantMessage("Checking.", (ToolCall("c1", "weather", '{"city": "Paris"}'),)),
            ToolResult("c1", "weather", "Sunny"),
            AssistantMessage("It's sunny"),
        )

        assert _input(history, ()) == [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Checking."},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": '{"city": "Paris"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "Sunny"},
            {"role": "assistant", "content": "It's sunny"},
        ]

    def it_sends_images_as_input_image_parts_in_messages_and_tool_results():
        picture = Image(b"\x89PNG", "image/png")
        history = (UserMessage("What is this?", (picture,)), ToolResult("c1", "snapshot", "[image]", images=(picture,)))

        assert _input(history, ()) == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw==", "detail": "auto"},
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "input_text", "text": "[image]"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="},
                ],
            },
        ]

    def it_sends_files_as_input_file_parts_in_messages_and_tool_results():
        paper = File(b"%PDF", name="paper.pdf")
        history = (UserMessage("Summarize [file]", files=(paper,)), ToolResult("c1", "pdf", "[file]", files=(paper,)))
        part = {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERg==", "filename": "paper.pdf"}

        assert _input(history, ()) == [
            {"role": "user", "content": [{"type": "input_text", "text": "Summarize [file]"}, part]},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "[file]"}, part],
            },
        ]

    def it_turns_a_search_and_its_result_into_the_tool_search_items_carrying_the_definitions():
        history = (
            AssistantMessage(tool_calls=(ToolCall("c1", "tool_search", '{"query": "weather"}'),)),
            ToolResult("c1", "tool_search", "Loaded", tools=("weather",)),
        )

        assert _input(history, (weather,)) == [
            {
                "type": "tool_search_call",
                "call_id": "c1",
                "arguments": {"query": "weather"},
                "execution": "client",
                "status": "completed",
            },
            {
                "type": "tool_search_output",
                "call_id": "c1",
                "execution": "client",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "weather",
                        "description": "Weather in a city.",
                        "parameters": weather.schema,
                        "strict": False,
                        "defer_loading": True,
                    }
                ],
            },
        ]

    def it_refuses_audio():
        with pytest.raises(TypeError, match="audio"):
            _input((UserMessage("hi", audio=(Audio(b"RIFF", "audio/wav"),)),), ())
        with pytest.raises(TypeError, match="audio"):
            _input((ToolResult("c1", "record", "", audio=(Audio(b"RIFF", "audio/wav"),)),), ())


def describe_streaming():
    async def it_emits_deltas_and_the_assembled_message():
        first_turn: list[ResponseStreamEvent] = [
            text_delta("Let me check."),
            added(call("fc1", "c1", "weather")),
            arguments_delta("fc1", '{"city": '),
            added(call("fc2", "c2", "time")),
            arguments_delta("fc1", '"Paris"}'),
            arguments_delta("fc2", "{}"),
            done(call("fc1", "c1", "weather", '{"city": "Paris"}')),
            done(call("fc2", "c2", "time", "{}")),
            completed(input=12, output=3),
        ]
        second_turn: list[ResponseStreamEvent] = [text_delta("done"), completed(input=20, output=1)]
        provider = Responses(temperature=0, max_tokens=300, reasoning_effort="low")
        client = Replaying(first_turn, second_turn)
        provider.__dict__["_client"] = client

        @agent(provider=provider, system="Be brief.", capabilities=[weather])
        async def bot(message: str) -> AsyncIterator[Event]: ...

        events = [e async for e in bot("weather?")]

        assert events == [
            TextDelta("Let me check."),
            ToolCallDelta("c1", "weather", '{"city": '),
            ToolCallDelta("c1", "weather", '"Paris"}'),
            ToolCallDelta("c2", "time", "{}"),
            AssistantMessage(
                "Let me check.",
                (ToolCall("c1", "weather", '{"city": "Paris"}'), ToolCall("c2", "time", "{}")),
                usage=Usage(12, 3),
            ),
            ToolResult("c1", "weather", "sunny in Paris"),
            ToolResult("c2", "time", "Unknown tool: time", is_error=True),
            TextDelta("done"),
            AssistantMessage("done", usage=Usage(20, 1)),
        ]
        first = client.calls[0]
        assert first["instructions"] == "Be brief."
        assert first["tools"] == [_tool(weather)]
        assert (first["temperature"], first["max_output_tokens"], first["reasoning"]) == (0, 300, {"effort": "low"})
        assert (first["store"], first["text"]) == (False, omit)
        second = client.calls[1]["input"]
        assert isinstance(second, list) and second[-2:] == [
            {"type": "function_call_output", "call_id": "c1", "output": "sunny in Paris"},
            {"type": "function_call_output", "call_id": "c2", "output": "Unknown tool: time"},
        ]

    async def it_runs_lazy_search_as_the_client_executed_tool_search_and_loads_through_its_output():
        search = ResponseToolSearchCall(
            id="ts1",
            call_id="c1",
            arguments={"query": "weather"},
            execution="client",
            status="completed",
            type="tool_search_call",
        )
        provider = Responses()
        client = Replaying([added(search), done(search)], [text_delta("done")])
        provider.__dict__["_client"] = client

        @agent(provider=provider, capabilities=[lazy(weather)])
        async def bot(message: str) -> str: ...

        assert await bot("weather?") == "done"

        tools = [_search(Search.over((weather,)))]
        assert tools[0]["type"] == "tool_search" and tools[0]["execution"] == "client"
        assert (client.calls[0]["tools"], client.calls[1]["tools"]) == (tools, tools)
        assert client.calls[1]["input"] == [
            {"role": "user", "content": "weather?"},
            {
                "type": "tool_search_call",
                "call_id": "c1",
                "arguments": {"query": "weather"},
                "execution": "client",
                "status": "completed",
            },
            {
                "type": "tool_search_output",
                "call_id": "c1",
                "execution": "client",
                "status": "completed",
                "tools": [_tool(weather, deferred=True)],
            },
        ]

    async def it_asks_for_structured_output_through_the_text_format():
        provider = Responses()
        client = Replaying([text_delta('{"value": {"city": "Paris", "sunny": true}}')])
        provider.__dict__["_client"] = client

        @agent(provider=provider)
        async def forecast(city: str) -> Forecast: ...

        assert await forecast("Paris") == Forecast(city="Paris", sunny=True)
        text = client.calls[0]["text"]
        assert isinstance(text, dict) and list(text) == ["format"]
        schema = text["format"].pop("schema")
        assert text["format"] == {"type": "json_schema", "name": "output"}
        assert schema["required"] == ["value"] and "Forecast" in schema["$defs"]

    async def it_refuses_media_answers_and_raises_on_refusal_or_failure():
        @agent(provider=Responses())
        async def speak(text: str) -> Audio: ...

        with pytest.raises(TypeError, match="text only"):
            await speak("hi")

        refusing, failing = Responses(), Responses()
        refusal = ResponseRefusalDoneEvent(
            content_index=0, item_id="m1", output_index=0, refusal="no", sequence_number=1, type="response.refusal.done"
        )
        error = ResponseError(code="server_error", message="boom")
        failure = ResponseFailedEvent(response=response(error=error), sequence_number=1, type="response.failed")
        refusing.__dict__["_client"] = Replaying([refusal])
        failing.__dict__["_client"] = Replaying([failure])

        @agent(provider=refusing)
        async def refused(message: str) -> str: ...

        @agent(provider=failing)
        async def failed(message: str) -> str: ...

        with pytest.raises(RuntimeError, match="refused the request: no"):
            await refused("?")
        with pytest.raises(RuntimeError, match="failed: boom"):
            await failed("?")


def call(id: str, call_id: str, name: str, arguments: str = "") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(id=id, call_id=call_id, name=name, arguments=arguments, type="function_call")


def added(item: ResponseFunctionToolCall | ResponseToolSearchCall) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent(item=item, output_index=0, sequence_number=0, type="response.output_item.added")


def done(item: ResponseFunctionToolCall | ResponseToolSearchCall) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent(item=item, output_index=0, sequence_number=0, type="response.output_item.done")


def text_delta(text: str) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        content_index=0,
        delta=text,
        item_id="m1",
        logprobs=[],
        output_index=0,
        sequence_number=0,
        type="response.output_text.delta",
    )


def arguments_delta(item_id: str, piece: str) -> ResponseFunctionCallArgumentsDeltaEvent:
    return ResponseFunctionCallArgumentsDeltaEvent(
        delta=piece, item_id=item_id, output_index=0, sequence_number=0, type="response.function_call_arguments.delta"
    )


def completed(*, input: int, output: int) -> ResponseCompletedEvent:
    usage = ResponseUsage(
        input_tokens=input,
        input_tokens_details=InputTokensDetails(cache_write_tokens=0, cached_tokens=0),
        output_tokens=output,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=input + output,
    )
    return ResponseCompletedEvent(response=response(usage=usage), sequence_number=0, type="response.completed")


def response(*, usage: ResponseUsage | None = None, error: ResponseError | None = None) -> Response:
    return Response(
        id="r",
        created_at=0,
        model="test",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        usage=usage,
        error=error,
    )


class Replaying:
    """Stands in for `AsyncOpenAI`: each `responses.create` records its arguments and streams the next turn."""

    def __init__(self, *turns: list[ResponseStreamEvent]) -> None:
        self.responses = SimpleNamespace(create=self.create)
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    async def create(self, **arguments: object) -> AsyncIterator[ResponseStreamEvent]:
        async def stream(events: list[ResponseStreamEvent]) -> AsyncIterator[ResponseStreamEvent]:
            for event in events:
                yield event

        self.calls.append(arguments)
        return stream(self.turns.pop(0))


class Forecast(BaseModel):
    city: str
    sunny: bool


@pytest.mark.integration
def describe_responses():
    async def it_calls_tools_and_returns_structured_output():
        @agent(provider=Responses(reasoning_effort="low"), capabilities=[weather])
        async def forecast(city: str) -> Forecast:
            """Use the weather tool to tell whether it is sunny in {city}."""
            ...

        assert await forecast("Paris") == Forecast(city="Paris", sunny=True)
