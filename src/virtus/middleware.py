from collections.abc import AsyncIterator, Callable, Sequence
from typing import Protocol, runtime_checkable

from virtus.message import Event
from virtus.request import Request

type Next = Callable[[Request], AsyncIterator[Event]]
type Middleware = Callable[[Request, Next], AsyncIterator[Event]]
"""Wraps one step of an agent run: a provider call followed by the tool calls it requested."""


@runtime_checkable
class Capability(Protocol):
    def middleware(self) -> Sequence[Middleware]: ...
