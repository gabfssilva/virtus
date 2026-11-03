from collections.abc import AsyncIterator, Mapping, Sequence
from functools import cached_property

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
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

from liteagent.message import AssistantMessage, Message, TextDelta, ToolCall, ToolResult, UserMessage
from liteagent.provider import Provider
from liteagent.request import Request
from liteagent.tool import Tool


class OpenAI(Provider):
    """Chat Completions API; `base_url` also covers compatible servers (OpenRouter, DeepSeek, Ollama)."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._base_url = base_url

    async def complete(self, request: Request) -> AsyncIterator[TextDelta | AssistantMessage]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages(request.system, request.messages),
            tools=[_tool(t) for t in request.tools] or omit,
            response_format=_response_format(request.output) if request.output else omit,
            temperature=omit if self.temperature is None else self.temperature,
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
                    fragments.setdefault(fragment.index, []).append(fragment)

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
            case UserMessage(content=content):
                return ChatCompletionUserMessageParam(role="user", content=content)
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
