"""The Realtime API over WebSocket, for `OpenAI.connect`; needs the `realtime` extra (`websockets`).

The protocol is OpenAI's; oMLX serves the same subset locally. Turns are closed by the client: the
server never detects speech here, so `turn_detection` is sent as null and the thread commits each turn.
"""

from base64 import b64decode, b64encode
from collections.abc import AsyncIterator, Sequence

from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
)
from openai.types.realtime.realtime_audio_config_output_param import RealtimeAudioConfigOutputParam
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_conversation_item_function_call_output_param import (
    RealtimeConversationItemFunctionCallOutputParam,
)
from openai.types.realtime.realtime_conversation_item_user_message_param import (
    Content,
    RealtimeConversationItemUserMessageParam,
)
from openai.types.realtime.realtime_function_tool_param import RealtimeFunctionToolParam
from openai.types.realtime.realtime_session_create_request_param import RealtimeSessionCreateRequestParam
from openai.types.realtime.realtime_tools_config_param import RealtimeToolsConfigParam

from virtus.message import (
    AssistantMessage,
    Audio,
    AudioDelta,
    Delta,
    Image,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    Usage,
)
from virtus.provider import Session
from virtus.tool import Tool

PCM = AudioPCM(type="audio/pcm", rate=24000)


class RealtimeSession(Session):
    def __init__(self, connection: AsyncRealtimeConnection, voice: str | None) -> None:
        self._connection = connection
        self._voice = voice
        self._configured: RealtimeSessionCreateRequestParam | None = None
        self._reply: str | None = None
        """Id of the assistant item being spoken, the only one the server lets us truncate."""

    async def open(self) -> None:
        """Turn server speech detection off before any audio arrives; OpenAI's sessions start with it on."""
        await self._configure("", ())

    async def hear(self, pcm: bytes) -> None:
        await self._connection.input_audio_buffer.append(audio=b64encode(pcm).decode())

    async def show(self, image: Image) -> None:
        url = f"data:{image.media_type};base64,{b64encode(image.data).decode()}"
        item = RealtimeConversationItemUserMessageParam(
            type="message", role="user", content=[Content(type="input_image", image_url=url)]
        )
        await self._connection.conversation.item.create(item=item)

    async def commit(self) -> None:
        await self._connection.input_audio_buffer.commit()

    async def respond(self, system: str, tools: Sequence[Tool]) -> AsyncIterator[Delta | AssistantMessage]:
        await self._configure(system, tools)
        await self._connection.response.create()

        usage: Usage | None = None
        transcript: list[str] = []
        sound: list[bytes] = []
        calls: list[ToolCall] = []

        async for event in self._connection:
            match event:
                case ResponseOutputItemAddedEvent(item=item) if item.type == "message" and item.id:
                    self._reply = item.id
                case ResponseAudioTranscriptDeltaEvent(delta=delta):
                    transcript.append(delta)
                    yield TextDelta(delta)
                case ResponseAudioDeltaEvent(delta=delta):
                    sound.append(b64decode(delta))
                    yield AudioDelta(sound[-1])
                case ResponseFunctionCallArgumentsDoneEvent(call_id=call_id, name=name, arguments=arguments):
                    calls.append(ToolCall(call_id, name, arguments))
                    yield ToolCallDelta(call_id, name, arguments)
                case RealtimeErrorEvent(error=error) if error.code != "response_cancel_not_active":
                    raise RuntimeError(f"realtime {error.code or error.type}: {error.message}")
                case ResponseDoneEvent(response=response):
                    if response.status == "failed":
                        raise RuntimeError(f"the reply failed: {response.status_details!r}")
                    if response.usage:
                        usage = Usage(response.usage.input_tokens or 0, response.usage.output_tokens or 0)
                    break
                case _:
                    pass

        spoken = Audio.pcm16(b"".join(sound)) if sound else None
        yield AssistantMessage("".join(transcript), tuple(calls), audio=spoken, usage=usage)

    async def deliver(self, result: ToolResult) -> None:
        item = RealtimeConversationItemFunctionCallOutputParam(
            type="function_call_output", call_id=result.call_id, output=result.content
        )
        await self._connection.conversation.item.create(item=item)

    async def interrupt(self, heard_ms: int) -> None:
        """The model finishes generating long before playback ends, so the cancel usually finds nothing to
        stop and the server says so; that error is ignored in `respond`. The truncation is what matters."""
        await self._connection.response.cancel()
        if self._reply:
            item = self._connection.conversation.item
            await item.truncate(item_id=self._reply, content_index=0, audio_end_ms=heard_ms)

    async def _configure(self, system: str, tools: Sequence[Tool]) -> None:
        session = self._session(system, tools)
        if session != self._configured:
            await self._connection.session.update(session=session)
            self._configured = session

    def _session(self, system: str, tools: Sequence[Tool]) -> RealtimeSessionCreateRequestParam:
        output = RealtimeAudioConfigOutputParam(format=PCM)
        if self._voice:
            output["voice"] = self._voice
        functions: RealtimeToolsConfigParam = [
            RealtimeFunctionToolParam(
                type="function", name=tool.name, description=tool.description, parameters=dict(tool.schema)
            )
            for tool in tools
        ]
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=system,
            output_modalities=["audio"],
            tools=functions,
            audio={"input": {"format": PCM, "turn_detection": None}, "output": output},
        )
