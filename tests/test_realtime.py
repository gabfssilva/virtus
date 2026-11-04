import io
import wave
from base64 import b64encode
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    RealtimeConversationItemAssistantMessage,
    RealtimeErrorEvent,
    RealtimeResponse,
    RealtimeServerEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
)
from openai.types.realtime.realtime_error import RealtimeError

from virtus import AssistantMessage, Audio, AudioDelta, Image, TextDelta, ToolCall, ToolCallDelta, agent, tool
from virtus.providers import OpenAI

PCM = {"type": "audio/pcm", "rate": 24000}


def describe_realtime_session():
    async def it_turns_server_speech_detection_off_before_any_audio():
        connection = Connection()
        provider = OpenAI("gpt-realtime-2.1-mini")
        provider.__dict__["_client"] = SimpleNamespace(realtime=SimpleNamespace(connect=connection.connect))

        @agent(provider=provider)
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live():
            pass

        [(kind, arguments)] = connection.sent
        assert kind == "session.update"
        assert arguments["session"] == {
            "type": "realtime",
            "instructions": "",
            "output_modalities": ["audio"],
            "tools": [],
            "audio": {"input": {"format": PCM, "turn_detection": None}, "output": {"format": PCM}},
        }

    async def it_configures_the_session_once_and_streams_the_spoken_reply():
        @tool
        def clock() -> str:
            """The time."""
            return "12:00"

        connection = Connection(
            [
                added("i1"),
                transcript("It is "),
                audio(b"\x01\x02"),
                transcript("noon"),
                audio(b"\x03\x04"),
                done("completed"),
            ],
            [added("i2"), transcript("Bye"), done("completed")],
        )
        provider = OpenAI("gpt-realtime-2.1-mini", voice="marin")
        provider.__dict__["_client"] = SimpleNamespace(realtime=SimpleNamespace(connect=connection.connect))

        @agent(provider=provider, system="Be brief.", capabilities=[clock])
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live() as call:
            await call.hear(b"\x00\x00")
            first = [e async for e in call.turn()]
            second = [e async for e in call.turn()]

        assert connection.models == ["gpt-realtime-2.1-mini"]
        assert [name for name, _ in connection.sent] == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "session.update",
            "response.create",
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert connection.sent[1][1] == {"audio": "AAA="}
        assert connection.sent[3][1] == {
            "session": {
                "type": "realtime",
                "instructions": "Be brief.",
                "output_modalities": ["audio"],
                "tools": [
                    {"type": "function", "name": "clock", "description": "The time.", "parameters": clock.schema}
                ],
                "audio": {
                    "input": {"format": PCM, "turn_detection": None},
                    "output": {"format": PCM, "voice": "marin"},
                },
            }
        }
        assert first == [
            TextDelta("It is "),
            AudioDelta(b"\x01\x02"),
            TextDelta("noon"),
            AudioDelta(b"\x03\x04"),
            AssistantMessage("It is noon", audio=Audio.pcm16(b"\x01\x02\x03\x04")),
        ]
        assert second[-1] == AssistantMessage("Bye", audio=None)

    async def it_runs_tool_calls_and_delivers_their_outputs():
        @tool
        def clock() -> str:
            return "12:00"

        connection = Connection(
            [added("i1"), function_call("c1", "clock", "{}"), done("completed")],
            [added("i2"), transcript("Noon."), done("completed")],
        )
        provider = OpenAI("gpt-realtime-2.1-mini")
        provider.__dict__["_client"] = SimpleNamespace(realtime=SimpleNamespace(connect=connection.connect))

        @agent(provider=provider, capabilities=[clock])
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live() as call:
            events = [e async for e in call.turn()]

        assert events[0] == ToolCallDelta("c1", "clock", "{}")
        assert events[1] == AssistantMessage(tool_calls=(ToolCall("c1", "clock", "{}"),))
        output = {"item": {"type": "function_call_output", "call_id": "c1", "output": "12:00"}}
        assert ("conversation.item.create", output) in connection.sent
        assert events[-1] == AssistantMessage("Noon.")
        assert "voice" not in repr(connection.sent[0])

    async def it_interrupts_the_reply_being_spoken_and_shows_images():
        connection = Connection([added("i1"), transcript("Long story"), done("cancelled")])
        provider = OpenAI("gpt-realtime-2.1-mini")
        provider.__dict__["_client"] = SimpleNamespace(realtime=SimpleNamespace(connect=connection.connect))

        @agent(provider=provider)
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live() as call:
            await call.show(Image(b"\x89PNG", "image/png"))
            [e async for e in call.turn()]
            await call.interrupt(750)

        assert connection.sent[1] == (
            "conversation.item.create",
            {
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="}],
                }
            },
        )
        assert connection.sent[-2:] == [
            ("response.cancel", {}),
            ("conversation.item.truncate", {"item_id": "i1", "content_index": 0, "audio_end_ms": 750}),
        ]

    async def it_raises_on_server_errors():
        detail = RealtimeError(type="invalid_request_error", code="unsupported", message="no")
        error = RealtimeErrorEvent(event_id="e", type="error", error=detail)
        connection = Connection([error])
        provider = OpenAI("gpt-realtime-2.1-mini")
        provider.__dict__["_client"] = SimpleNamespace(realtime=SimpleNamespace(connect=connection.connect))

        @agent(provider=provider)
        async def assistant(message: str) -> str: ...

        with pytest.raises(RuntimeError, match="unsupported: no"):
            async with assistant.thread().live() as call:
                [e async for e in call.turn()]


def it_wraps_pcm16_as_a_wav_file():
    clip = Audio.pcm16(b"\x01\x02\x03\x04")

    with wave.open(io.BytesIO(clip.data)) as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 24_000)
        assert file.readframes(2) == b"\x01\x02\x03\x04"
    assert clip.media_type == "audio/wav"


def added(item_id: str) -> RealtimeServerEvent:
    item = RealtimeConversationItemAssistantMessage(id=item_id, type="message", role="assistant", content=[])
    return ResponseOutputItemAddedEvent(
        event_id="e", type="response.output_item.added", response_id="r", output_index=0, item=item
    )


def transcript(delta: str) -> RealtimeServerEvent:
    return ResponseAudioTranscriptDeltaEvent(
        event_id="e",
        type="response.output_audio_transcript.delta",
        response_id="r",
        item_id="i",
        output_index=0,
        content_index=0,
        delta=delta,
    )


def audio(pcm: bytes) -> RealtimeServerEvent:
    return ResponseAudioDeltaEvent(
        event_id="e",
        type="response.output_audio.delta",
        response_id="r",
        item_id="i",
        output_index=0,
        content_index=0,
        delta=b64encode(pcm).decode(),
    )


def function_call(call_id: str, name: str, arguments: str) -> RealtimeServerEvent:
    return ResponseFunctionCallArgumentsDoneEvent(
        event_id="e",
        type="response.function_call_arguments.done",
        response_id="r",
        item_id="f",
        output_index=1,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def done(status: str) -> RealtimeServerEvent:
    response = RealtimeResponse.model_validate({"id": "r", "status": status})
    return ResponseDoneEvent(event_id="e", type="response.done", response=response)


class Connection:
    """Stands in for `AsyncRealtimeConnection`: records client events, replays one list of server events per reply."""

    def __init__(self, *responses: list[RealtimeServerEvent]) -> None:
        self.responses = list(responses)
        self.sent: list[tuple[str, dict[str, object]]] = []
        self.models: list[str] = []
        self.session = SimpleNamespace(update=self._sender("session.update"))
        self.response = SimpleNamespace(create=self._sender("response.create"), cancel=self._sender("response.cancel"))
        self.input_audio_buffer = SimpleNamespace(
            append=self._sender("input_audio_buffer.append"), commit=self._sender("input_audio_buffer.commit")
        )
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(
                create=self._sender("conversation.item.create"), truncate=self._sender("conversation.item.truncate")
            )
        )

    @asynccontextmanager
    async def connect(self, model: str) -> AsyncIterator["Connection"]:
        self.models.append(model)
        yield self

    def __aiter__(self) -> AsyncIterator[RealtimeServerEvent]:
        async def events() -> AsyncIterator[RealtimeServerEvent]:
            for event in self.responses.pop(0):
                yield event

        return events()

    def _sender(self, kind: str):
        async def send(**arguments: object) -> None:
            self.sent.append((kind, arguments))

        return send

