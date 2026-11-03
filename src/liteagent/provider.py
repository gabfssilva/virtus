from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from liteagent.message import AssistantMessage, TextDelta
from liteagent.request import Request


@runtime_checkable
class Provider(Protocol):
    def complete(self, request: Request) -> AsyncIterator[TextDelta | AssistantMessage]:
        """Stream the reply as text deltas, ending with exactly one complete `AssistantMessage`."""
        ...
