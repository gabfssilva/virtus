from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

from virtus.message import AssistantMessage, Event, Message, ToolResult, UserMessage
from virtus.middleware import Middleware, Next
from virtus.provider import Provider
from virtus.request import Request

SUMMARY_PROMPT = (
    "Summarize the conversation below so the summary can replace it. Keep facts, decisions, results "
    "of tool calls, open questions and anything the user asked to remember. Be concise."
)


SUMMARY = "Summary of the earlier conversation:"


def compaction(
    max_tokens: int = 32_000,
    keep: int = 8,
    *,
    chars_per_token: int = 4,
    provider: Provider | None = None,
    prompt: str = SUMMARY_PROMPT,
    header: str = SUMMARY,
) -> Middleware:
    """Replace older messages with a summary once the conversation exceeds `max_tokens`.

    Tokens are estimated as `chars_per_token` characters each. At least the last `keep` messages stay
    verbatim; the boundary moves back so a tool result is never separated from the call that produced it.
    The summary is written by `provider`, by default the request's, following `prompt`, and goes in as a
    user message opened by `header`. Like any rewrite of the messages it becomes the history, so it is not
    recomputed until the conversation grows past the limit again.
    """

    async def summarize(messages: Sequence[Message], request: Request) -> str:
        summary = replace(request, system=prompt, messages=(UserMessage(_transcript(messages)),), tools=(), output=None)
        replies = [e async for e in (provider or request.provider).complete(summary) if isinstance(e, AssistantMessage)]
        return replies[-1].content

    async def compact(request: Request, next: Next) -> AsyncIterator[Event]:
        messages = request.messages
        start = max(
            (i for i in range(1, len(messages) - keep + 1) if not isinstance(messages[i], ToolResult)),
            default=0,
        )

        if start and len(_transcript(messages)) // chars_per_token > max_tokens:
            summary = UserMessage(f"{header}\n\n{await summarize(messages[:start], request)}")
            request = replace(request, messages=(summary, *messages[start:]))

        async for event in next(request):
            yield event

    return compact


def _transcript(messages: Sequence[Message]) -> str:
    def lines(message: Message) -> list[str]:
        match message:
            case UserMessage(content=content, images=images, audio=audio, files=files):
                return [f"user: {content}{_media(images, audio, files)}"]
            case AssistantMessage(content=content, tool_calls=calls, images=images, audio=audio):
                text = [f"assistant: {content}{_media(images, (audio,) if audio else ())}"] if content or images else []
                return [*text, *(f"assistant called {c.name}({c.arguments})" for c in calls)]
            case ToolResult(name=name, content=content, images=images, audio=audio, files=files):
                return [f"{name} returned: {content}{_media(images, audio, files)}"]

    return "\n".join(line for message in messages for line in lines(message))


def _media(images: Sequence[object], audio: Sequence[object], files: Sequence[object] = ()) -> str:
    counts = ((images, "image(s)"), (audio, "audio clip(s)"), (files, "file(s)"))
    notes = [f"{len(items)} {kind}" for items, kind in counts if items]
    return f" [{', '.join(notes)}]" if notes else ""
