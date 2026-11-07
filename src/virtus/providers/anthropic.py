import json
from base64 import b64encode
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from functools import cached_property
from itertools import groupby
from typing import Literal

from anthropic import AsyncAnthropic, omit
from anthropic.types import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from virtus.message import (
    AssistantMessage,
    Delta,
    Image,
    Message,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    Usage,
    UserMessage,
)
from virtus.provider import Provider
from virtus.request import Request
from virtus.tool import Tool

type Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Anthropic(Provider):
    """Messages API with streaming; `effort` trades reasoning depth for speed and cost."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16_000,
        effort: Effort | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._api_key = api_key
        self._base_url = base_url

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        if request.media:
            raise TypeError(f"{self!r}: the Messages API answers in text only, not {', '.join(request.media)}")

        stream = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=request.system or omit,
            messages=_messages(request.messages),
            tools=[_tool(t) for t in request.tools] or omit,
            output_config=self._output_config(request.output) or omit,
            stream=True,
        )

        usage: Usage | None = None
        text: list[str] = []
        calls: dict[int, tuple[str, str, list[str]]] = {}
        """Open tool_use blocks by index: id, name and the argument fragments so far."""

        async for event in stream:
            match event.type:
                case "message_start":
                    usage = Usage(event.message.usage.input_tokens, event.message.usage.output_tokens)
                case "content_block_start" if event.content_block.type == "tool_use":
                    calls[event.index] = (event.content_block.id, event.content_block.name, [])
                case "content_block_delta" if event.delta.type == "text_delta":
                    text.append(event.delta.text)
                    yield TextDelta(event.delta.text)
                case "content_block_delta" if event.delta.type == "input_json_delta":
                    id, name, fragments = calls[event.index]
                    fragments.append(event.delta.partial_json)
                    yield ToolCallDelta(id, name, event.delta.partial_json)
                case "message_delta" if event.delta.stop_reason == "refusal":
                    details = event.delta.stop_details
                    raise RuntimeError(f"{self!r} refused the request: {details.explanation if details else ''}")
                case "message_delta":
                    usage = Usage(event.usage.input_tokens or (usage.input if usage else 0), event.usage.output_tokens)
                case _:
                    pass

        yield AssistantMessage(
            "".join(text),
            tuple(ToolCall(id, name, "".join(fragments) or "{}") for _, (id, name, fragments) in sorted(calls.items())),
            usage=usage,
        )

    def _output_config(self, schema: Mapping[str, object] | None) -> OutputConfigParam:
        config: OutputConfigParam = {}
        if self.effort is not None:
            config["effort"] = self.effort
        if schema is not None:
            config["format"] = {"type": "json_schema", "schema": dict(schema)}
        return config

    @cached_property
    def _client(self) -> AsyncAnthropic:
        """Created on first use, so building an agent doesn't require credentials."""
        return AsyncAnthropic(api_key=self._api_key, base_url=self._base_url)

    def __repr__(self) -> str:
        return f"Anthropic({self.model!r})"


def _messages(history: Sequence[Message]) -> list[MessageParam]:
    """Consecutive tool results share one user message, as the API expects for parallel calls."""

    def convert(message: Message) -> MessageParam:
        match message:
            case UserMessage(audio=audio) if audio:
                raise TypeError("the Messages API does not take audio")
            case UserMessage(content=content, images=()):
                return {"role": "user", "content": content}
            case UserMessage(content=content, images=images):
                blocks: list[ContentBlockParam] = [*map(_image, images), TextBlockParam(type="text", text=content)]
                return {"role": "user", "content": blocks}
            case AssistantMessage(content=content, tool_calls=calls):
                text: list[ContentBlockParam] = [TextBlockParam(type="text", text=content)] if content else []
                uses: list[ContentBlockParam] = [
                    ToolUseBlockParam(type="tool_use", id=c.id, name=c.name, input=json.loads(c.arguments or "{}"))
                    for c in calls
                ]
                return {"role": "assistant", "content": [*text, *uses]}
            case ToolResult():
                raise AssertionError("tool results are grouped before conversion")

    def result(message: ToolResult) -> ContentBlockParam:
        if message.audio:
            raise TypeError("the Messages API does not take audio")
        content: str | list[TextBlockParam | ImageBlockParam] = message.content
        if message.images:
            content = [TextBlockParam(type="text", text=message.content), *map(_image, message.images)]
        return ToolResultBlockParam(
            type="tool_result", tool_use_id=message.call_id, content=content, is_error=message.is_error
        )

    def grouped() -> Iterator[MessageParam]:
        for are_results, group in groupby(history, key=lambda m: isinstance(m, ToolResult)):
            if are_results:
                yield {"role": "user", "content": [result(m) for m in group if isinstance(m, ToolResult)]}
            else:
                yield from (convert(m) for m in group)

    return list(grouped())


def _image(image: Image) -> ImageBlockParam:
    source = Base64ImageSourceParam(type="base64", media_type=image.media_type, data=b64encode(image.data).decode())
    return ImageBlockParam(type="image", source=source)


def _tool(tool: Tool) -> ToolParam:
    return {"name": tool.name, "description": tool.description, "input_schema": dict(tool.schema)}

