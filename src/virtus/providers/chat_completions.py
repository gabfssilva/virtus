from base64 import b64decode, b64encode
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import cached_property
from itertools import groupby
from typing import Literal

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudioParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from openai.types.chat.chat_completion_content_part_param import File as FilePart
from openai.types.chat.chat_completion_content_part_param import FileFile
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, TypeAdapter

from virtus.message import (
    AssistantMessage,
    Audio,
    AudioDelta,
    Delta,
    File,
    Image,
    Message,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    Usage,
    UserMessage,
    transcribed,
)
from virtus.provider import LiveProvider, Provider, Session
from virtus.request import Request
from virtus.tool import Tool


class ChatCompletions(Provider, LiveProvider):
    """Chat Completions API; `base_url` also covers compatible servers (OpenRouter, DeepSeek, Ollama, oMLX).

    Options left `None` are not sent, so the server's defaults apply. `reasoning_effort` only
    applies to reasoning models. `voice` speaks when the agent's answer is `Audio`, which only audio
    models accept; audio in the input needs one too. Spoken answers come back as 24 kHz mono WAV.
    `files=False` sends a `File`'s text instead of the file, for servers whose models read no PDFs.
    With a realtime model, `connect` opens the Realtime API for `Thread.live`.
    """

    def __init__(
        self,
        model: str = "gpt-6-luna",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600,
        max_retries: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = None,
        voice: str | None = None,
        files: bool = True,
    ) -> None:
        self.model = model
        self.files = files
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.voice = voice
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        wants_audio = "audio" in request.media
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=_messages(request.system, request.messages, files=self.files),
            # Chat Completions has no deferred loading: a deferred tool is sent once a result has loaded it.
            tools=[_tool(t) for t in (*request.tools, *request.loaded)] or omit,
            response_format=_response_format(request.output) if request.output else omit,
            temperature=omit if self.temperature is None else self.temperature,
            max_completion_tokens=omit if self.max_tokens is None else self.max_tokens,
            reasoning_effort=omit if self.reasoning_effort is None else self.reasoning_effort,
            # Streamed audio is only offered as raw PCM; it is wrapped as WAV below.
            audio=ChatCompletionAudioParam(voice=self.voice or "alloy", format="pcm16") if wants_audio else omit,
            # The SDK's `modalities` does not admit `image`, which OpenRouter and Gemini-style servers take.
            extra_body={"modalities": ["text", *request.media]} if request.media else None,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage: Usage | None = None
        text: list[str] = []
        sound: list[bytes] = []
        images: list[Image] = []
        fragments: dict[int, list[ChoiceDeltaToolCall]] = {}

        async for chunk in stream:
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    yield TextDelta(delta.content)
                if delta.audio and delta.audio.transcript:
                    text.append(delta.audio.transcript)
                    yield TextDelta(delta.audio.transcript)
                if delta.audio and delta.audio.data:
                    sound.append(b64decode(delta.audio.data))
                    yield AudioDelta(sound[-1])
                images.extend(_images(delta))
                for fragment in delta.tool_calls or ():
                    parts = fragments.setdefault(fragment.index, [])
                    parts.append(fragment)
                    call = _tool_call(parts)
                    piece = fragment.function.arguments if fragment.function else None
                    yield ToolCallDelta(call.id, call.name, piece or "")

        yield AssistantMessage(
            "".join(text),
            tuple(_tool_call(f) for _, f in sorted(fragments.items())),
            tuple(images),
            Audio.pcm16(b"".join(sound)) if sound else None,
            usage,
        )

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Session]:
        """A Realtime API session on `model`, which must be a realtime model; needs the `realtime` extra."""
        from virtus.providers.openai_realtime import RealtimeSession

        async with self._client.realtime.connect(model=self.model) as connection:
            session = RealtimeSession(connection, self.voice)
            await session.open()
            yield session

    @cached_property
    def _client(self) -> AsyncOpenAI:
        """Created on first use, so building an agent doesn't require credentials."""
        return AsyncOpenAI(
            api_key=self._api_key, base_url=self._base_url, timeout=self._timeout, max_retries=self._max_retries
        )

    def __repr__(self) -> str:
        return f"ChatCompletions({self.model!r})"


def _messages(system: str, history: Sequence[Message], *, files: bool = True) -> list[ChatCompletionMessageParam]:
    """A `tool` message carries text only, so media returned by a run of tool calls follows it as a user message.

    A spoken answer goes back as its transcript, and an answer's images are not sent back at all: the
    API has no place for either in an assistant message. Without `files`, a file goes as its text.
    """

    def readable(message: Message) -> Message:
        match message:
            case UserMessage(content=content, files=attached) | ToolResult(content=content, files=attached) if (
                attached and not files
            ):
                return replace(message, content=transcribed(content, attached), files=())
            case _:
                return message

    def convert(message: Message) -> ChatCompletionMessageParam:
        match message:
            case UserMessage(content=content, images=(), audio=(), files=()):
                return ChatCompletionUserMessageParam(role="user", content=content)
            case UserMessage(content=content, images=images, audio=audio, files=attached):
                text = ChatCompletionContentPartTextParam(type="text", text=content)
                media = _media(images, audio, attached)
                return ChatCompletionUserMessageParam(role="user", content=[text, *media])
            case AssistantMessage(content=content, tool_calls=()):
                return ChatCompletionAssistantMessageParam(role="assistant", content=content)
            case AssistantMessage(content=content, tool_calls=calls):
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=content or None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCallParam(
                            id=c.id, type="function", function=Function(name=c.name, arguments=c.arguments)
                        )
                        for c in calls
                    ],
                )
            case ToolResult(call_id=call_id, content=content):
                return ChatCompletionToolMessageParam(role="tool", tool_call_id=call_id, content=content)

    def grouped() -> Iterator[ChatCompletionMessageParam]:
        for are_results, group in groupby(map(readable, history), key=lambda m: isinstance(m, ToolResult)):
            messages = list(group)
            yield from map(convert, messages)
            results = [m for m in messages if isinstance(m, ToolResult)]
            images = [i for m in results for i in m.images]
            audio = [a for m in results for a in m.audio]
            attached = [f for m in results for f in m.files]
            if are_results and (images or audio or attached):
                note = ChatCompletionContentPartTextParam(type="text", text="Media returned by the tool calls above.")
                yield ChatCompletionUserMessageParam(role="user", content=[note, *_media(images, audio, attached)])

    prompt = [ChatCompletionSystemMessageParam(role="system", content=system)] if system else []
    return [*prompt, *grouped()]


def _media(
    images: Sequence[Image], audio: Sequence[Audio], files: Sequence[File]
) -> list[ChatCompletionContentPartParam]:
    return [*map(_image, images), *map(_audio, audio), *map(_file, files)]


def _file(file: File) -> FilePart:
    data = f"data:{file.media_type};base64,{b64encode(file.data).decode()}"
    return FilePart(type="file", file=FileFile(file_data=data, filename=file.name))


def _image(image: Image) -> ChatCompletionContentPartImageParam:
    data = b64encode(image.data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{data}"}}


def _audio(audio: Audio) -> ChatCompletionContentPartInputAudioParam:
    match audio.media_type:
        case "audio/wav":
            format: Literal["wav", "mp3"] = "wav"
        case "audio/mpeg":
            format = "mp3"
        case other:
            raise TypeError(f"Chat Completions takes audio input as wav or mp3, not {other}")
    return {"type": "input_audio", "input_audio": {"data": b64encode(audio.data).decode(), "format": format}}


class _ImageUrl(BaseModel):
    url: str


class _ImagePart(BaseModel):
    image_url: _ImageUrl


_IMAGE_PARTS = TypeAdapter(list[_ImagePart])


def _images(delta: ChoiceDelta) -> Iterator[Image]:
    """Images generated by the model arrive as `images` on the delta, a data URL each; the SDK leaves them untyped."""
    for part in _IMAGE_PARTS.validate_python((delta.model_extra or {}).get("images") or []):
        header, _, data = part.image_url.url.partition(",")
        match header.removeprefix("data:").removesuffix(";base64"):
            case "image/jpeg" | "image/png" | "image/gif" | "image/webp" as media_type:
                yield Image(b64decode(data), media_type)
            case other:
                raise TypeError(f"unsupported image type in the answer: {other!r}")


def _tool(tool: Tool) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": dict(tool.schema)},
    }


def _response_format(schema: Mapping[str, object]) -> ResponseFormat:
    return {"type": "json_schema", "json_schema": {"name": "output", "schema": dict(schema)}}


def _tool_call(fragments: Sequence[ChoiceDeltaToolCall]) -> ToolCall:
    """Tool calls arrive split across chunks: the id and name once, the arguments in pieces."""
    functions = [f.function for f in fragments if f.function]
    return ToolCall(
        id=next(f.id for f in fragments if f.id),
        name="".join(f.name for f in functions if f.name),
        arguments="".join(f.arguments for f in functions if f.arguments),
    )
