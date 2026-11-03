from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

from virtus.filter import Filter, Next
from virtus.message import AssistantMessage, Event, Message, ToolResult, UserMessage
from virtus.request import Request

SUMMARY_PROMPT = (
    "Summarize the conversation below so the summary can replace it. Keep facts, decisions, results "
    "of tool calls, open questions and anything the user asked to remember. Be concise."
)


def compaction(max_tokens: int = 32_000, keep: int = 8) -> Filter:
    """Replace older messages with a summary once the conversation exceeds `max_tokens`.

    Tokens are estimated as four characters each. At least the last `keep` messages stay verbatim;
    the boundary moves back so a tool result is never separated from the call that produced it.
    The summary is written by the request's provider and, like any rewrite of the messages,
    becomes the history, so it is not recomputed until the conversation grows past the limit again.
    """

    async def summarize(messages: Sequence[Message], request: Request) -> str:
        summary = replace(
            request, system=SUMMARY_PROMPT, messages=(UserMessage(_transcript(messages)),), tools=(), output=None
        )
        replies = [e async for e in request.provider.complete(summary) if isinstance(e, AssistantMessage)]
        return replies[-1].content

    async def compact(request: Request, next: Next) -> AsyncIterator[Event]:
        messages = request.messages
        start = max(
            (i for i in range(1, len(messages) - keep + 1) if not isinstance(messages[i], ToolResult)),
            default=0,
        )

        if start and len(_transcript(messages)) // 4 > max_tokens:
            summary = UserMessage(
                f"Summary of the earlier conversation:\n\n{await summarize(messages[:start], request)}"
            )
            request = replace(request, messages=(summary, *messages[start:]))

        async for event in next(request):
            yield event

    return compact


def _transcript(messages: Sequence[Message]) -> str:
    def lines(message: Message) -> list[str]:
        match message:
            case UserMessage(content=content, images=images):
                return [f"user: {content}" + (f" [{len(images)} image(s)]" if images else "")]
            case AssistantMessage(content=content, tool_calls=calls):
                text = [f"assistant: {content}"] if content else []
                return [*text, *(f"assistant called {c.name}({c.arguments})" for c in calls)]
            case ToolResult(name=name, content=content):
                return [f"{name} returned: {content}"]

    return "\n".join(line for message in messages for line in lines(message))
