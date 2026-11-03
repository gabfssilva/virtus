from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from virtus.message import AssistantMessage, Delta
from virtus.request import Request


@runtime_checkable
class Provider(Protocol):
    def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        """Stream the reply as text and tool call deltas, ending with exactly one complete `AssistantMessage`."""
        ...
