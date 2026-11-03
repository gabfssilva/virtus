import json
import re
from collections.abc import AsyncIterator

from liteagent import AssistantMessage, Provider, Request, TextDelta, ToolCall


class Scripted(Provider):
    """Replies with the given messages in order and records every request it receives."""

    def __init__(self, *replies: AssistantMessage) -> None:
        self.replies = list(replies)
        self.requests: list[Request] = []

    async def complete(self, request: Request) -> AsyncIterator[TextDelta | AssistantMessage]:
        self.requests.append(request)
        reply = self.replies.pop(0)
        for word in re.findall(r"\S+\s*", reply.content):
            yield TextDelta(word)
        yield reply


def calls(*requested: tuple[str, dict[str, object]]) -> AssistantMessage:
    return AssistantMessage(
        tool_calls=tuple(ToolCall(f"call-{i}", name, json.dumps(args)) for i, (name, args) in enumerate(requested))
    )
