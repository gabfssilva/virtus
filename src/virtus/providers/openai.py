from base64 import b64encode
from collections.abc import AsyncIterator, Mapping, Sequence
from functools import cached_property

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared.reasoning_effort import ReasoningEffort

from virtus.message import (
    AssistantMessage,
    Delta,
    Image,
    Message,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    UserMessage,
)
from virtus.provider import Provider
from virtus.request import Request
from virtus.tool import Tool


class OpenAI(Provider):
    """Chat Completions API; `base_url` also covers compatible servers (OpenRouter, DeepSeek, Ollama).

    Options left `None` are not sent, so the server's defaults apply. `reasoning_effort` only
    applies to reasoning models.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
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

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages(request.system, request.messages),
            tools=[_tool(t) for t in request.tools] or omit,
            response_format=_response_format(request.output) if request.output else omit,
            temperature=omit if self.temperature is None else self.temperature,
            max_completion_tokens=omit if self.max_tokens is None else self.max_tokens,
            reasoning_effort=omit if self.reasoning_effort is None else self.reasoning_effort,
            stream=True,
        )

        text: list[str] = []
        fragments: dict[int, list[ChoiceDeltaToolCall]] = {}

        async for chunk in stream:
            for choice in chunk.choices:
                if choice.delta.content:
                    text.append(choice.delta.content)
                    yield TextDelta(choice.delta.content)
                for fragment in choice.delta.tool_calls or ():
                    parts = fragments.setdefault(fragment.index, [])
                    parts.append(fragment)
                    call = _tool_call(parts)
                    piece = fragment.function.arguments if fragment.function else None
                    yield ToolCallDelta(call.id, call.name, piece or "")

        yield AssistantMessage("".join(text), tuple(_tool_call(f) for _, f in sorted(fragments.items())))

    @cached_property
    def _client(self) -> AsyncOpenAI:
        """Created on first use, so building an agent doesn't require credentials."""
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    def __repr__(self) -> str:
        return f"OpenAI({self.model!r})"


def messages(system: str, history: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    def convert(message: Message) -> ChatCompletionMessageParam:
        match message:
            case UserMessage(content=content, images=()):
                return ChatCompletionUserMessageParam(role="user", content=content)
            case UserMessage(content=content, images=images):
                parts = [ChatCompletionContentPartTextParam(type="text", text=content), *map(_image, images)]
                return ChatCompletionUserMessageParam(role="user", content=parts)
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

    prompt = [ChatCompletionSystemMessageParam(role="system", content=system)] if system else []
    return [*prompt, *map(convert, history)]


def _image(image: Image) -> ChatCompletionContentPartImageParam:
    data = b64encode(image.data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{data}"}}


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
