import json
from base64 import b64encode
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from functools import cached_property

from openai import AsyncOpenAI, omit
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFormatTextConfigParam,
    ResponseFunctionCallOutputItemListParam,
    ResponseFunctionToolCallParam,
    ResponseInputFileContentParam,
    ResponseInputFileParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseTextConfigParam,
    ResponseToolSearchOutputItemParamParam,
    ToolParam,
    ToolSearchToolParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput, ToolSearchCall
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning

from virtus.lazy import SEARCH
from virtus.message import (
    AssistantMessage,
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
)
from virtus.provider import Provider
from virtus.request import Request
from virtus.tool import Tool


class Responses(Provider):
    """The Responses API with streaming, text, images and PDFs in, text out; the history goes in full (`store=False`).

    Deferred tools are loaded natively: `lazy()`'s search runs here as the API's client-executed `tool_search`,
    and what it loads travels as `tool_search_output` items carrying the definitions, so the request keeps
    only the tools in use. Options left `None` are not sent, so the server's defaults apply.
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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        if request.media:
            raise TypeError(f"{self!r}: the Responses API answers in text only, not {', '.join(request.media)}")

        stream = await self._client.responses.create(
            model=self.model,
            instructions=request.system or omit,
            input=_input(request.messages, request.deferred),
            tools=[_search(t) if t.name == SEARCH else _tool(t) for t in request.tools] or omit,
            text=ResponseTextConfigParam(format=_format(request.output)) if request.output else omit,
            temperature=omit if self.temperature is None else self.temperature,
            max_output_tokens=omit if self.max_tokens is None else self.max_tokens,
            reasoning=omit if self.reasoning_effort is None else Reasoning(effort=self.reasoning_effort),
            store=False,
            stream=True,
        )

        usage: Usage | None = None
        text: list[str] = []
        calls: list[ToolCall] = []
        started: dict[str, tuple[str, str]] = {}
        """Function calls in progress by item id: the call id and name their argument deltas belong to."""

        async for event in stream:
            match event.type:
                case "response.output_text.delta":
                    text.append(event.delta)
                    yield TextDelta(event.delta)
                case "response.output_item.added" if event.item.type == "function_call":
                    started[event.item.id or event.item.call_id] = (event.item.call_id, event.item.name)
                case "response.function_call_arguments.delta":
                    id, name = started[event.item_id]
                    yield ToolCallDelta(id, name, event.delta)
                case "response.output_item.done" if event.item.type == "function_call":
                    calls.append(ToolCall(event.item.call_id, event.item.name, event.item.arguments or "{}"))
                case "response.output_item.done" if event.item.type == "tool_search_call":
                    id, arguments = event.item.call_id or event.item.id, json.dumps(event.item.arguments)
                    calls.append(ToolCall(id, SEARCH, arguments))
                case "response.refusal.done":
                    raise RuntimeError(f"{self!r} refused the request: {event.refusal}")
                case "response.failed":
                    error = event.response.error
                    raise RuntimeError(f"{self!r} failed: {error.message if error else 'no error reported'}")
                case "response.completed" | "response.incomplete" if event.response.usage:
                    usage = Usage(event.response.usage.input_tokens, event.response.usage.output_tokens)
                case _:
                    pass

        yield AssistantMessage("".join(text), tuple(calls), usage=usage)

    @cached_property
    def _client(self) -> AsyncOpenAI:
        """Created on first use, so building an agent doesn't require credentials."""
        return AsyncOpenAI(
            api_key=self._api_key, base_url=self._base_url, timeout=self._timeout, max_retries=self._max_retries
        )

    def __repr__(self) -> str:
        return f"Responses({self.model!r})"


def _input(history: Sequence[Message], deferred: Sequence[Tool]) -> list[ResponseInputItemParam]:
    """The history as input items; a search call and its result become the API's own tool search items.

    The result's `tool_search_output` carries the definitions of the tools it loaded, found in `deferred` by
    name: that is how the API learns them, so loaded tools are never in `tools`. Its text has no place there.
    """

    def convert(message: Message) -> Iterator[ResponseInputItemParam]:
        match message:
            case UserMessage(audio=audio) | ToolResult(audio=audio) if audio:
                raise TypeError("the Responses API does not take audio")
            case UserMessage(content=content, images=(), files=()):
                yield EasyInputMessageParam(role="user", content=content)
            case UserMessage(content=content, images=images, files=files):
                text = ResponseInputTextParam(type="input_text", text=content)
                yield EasyInputMessageParam(role="user", content=[text, *map(_image, images), *map(_file, files)])
            case AssistantMessage(content=content, tool_calls=calls):
                if content:
                    yield EasyInputMessageParam(role="assistant", content=content)
                for call in calls:
                    yield _call(call)
            case ToolResult(call_id=call_id, name=name, tools=names) if name == SEARCH:
                tools = [_tool(t, deferred=True) for t in deferred if t.name in names]
                yield ResponseToolSearchOutputItemParamParam(
                    type="tool_search_output", call_id=call_id, execution="client", status="completed", tools=tools
                )
            case ToolResult(call_id=call_id, content=content, images=(), files=()):
                yield FunctionCallOutput(type="function_call_output", call_id=call_id, output=content)
            case ToolResult(call_id=call_id, content=content, images=images, files=files):
                parts: ResponseFunctionCallOutputItemListParam = [
                    ResponseInputTextContentParam(type="input_text", text=content),
                    *(ResponseInputImageContentParam(type="input_image", image_url=_url(i)) for i in images),
                    *(
                        ResponseInputFileContentParam(type="input_file", file_data=_url(f), filename=f.name)
                        for f in files
                    ),
                ]
                yield FunctionCallOutput(type="function_call_output", call_id=call_id, output=parts)

    return [item for message in history for item in convert(message)]


def _call(call: ToolCall) -> ResponseInputItemParam:
    if call.name == SEARCH:
        arguments: object = json.loads(call.arguments or "{}")
        return ToolSearchCall(
            type="tool_search_call", call_id=call.id, arguments=arguments, execution="client", status="completed"
        )
    return ResponseFunctionToolCallParam(
        type="function_call", call_id=call.id, name=call.name, arguments=call.arguments or "{}"
    )


def _image(image: Image) -> ResponseInputImageParam:
    return ResponseInputImageParam(type="input_image", image_url=_url(image), detail="auto")


def _file(file: File) -> ResponseInputFileParam:
    return ResponseInputFileParam(type="input_file", file_data=_url(file), filename=file.name)


def _url(media: Image | File) -> str:
    return f"data:{media.media_type};base64,{b64encode(media.data).decode()}"


def _tool(tool: Tool, *, deferred: bool = False) -> FunctionToolParam:
    definition = FunctionToolParam(
        type="function", name=tool.name, description=tool.description, parameters=dict(tool.schema), strict=False
    )
    if deferred:
        definition["defer_loading"] = True
    return definition


def _search(tool: Tool) -> ToolParam:
    """`lazy()`'s search tool as the API's client-executed `tool_search`; the model calls it by type, not by name."""
    return ToolSearchToolParam(
        type="tool_search", execution="client", description=tool.description, parameters=dict(tool.schema)
    )


def _format(schema: Mapping[str, object]) -> ResponseFormatTextConfigParam:
    return {"type": "json_schema", "name": "output", "schema": dict(schema)}
