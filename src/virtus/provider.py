from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from virtus.message import AssistantMessage, TextDelta
from virtus.request import Request


@runtime_checkable
class Provider(Protocol):
    def complete(self, request: Request) -> AsyncIterator[TextDelta | AssistantMessage]:
        """Stream the reply as text deltas, ending with exactly one complete `AssistantMessage`."""
        ...
