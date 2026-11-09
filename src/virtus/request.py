from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from virtus.message import Message, ToolResult

if TYPE_CHECKING:
    from virtus.provider import Provider
    from virtus.thread import Conversation
    from virtus.tool import Tool

type Modality = Literal["audio", "image"]


@dataclass(frozen=True)
class Request:
    """One call to the provider, as seen and rewritten by middleware.

    The `messages` that reach the provider become the conversation history, so a middleware that
    rewrites them (redaction, compaction) changes what is remembered, not only what is sent.
    """

    system: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    output: Mapping[str, object] | None
    """JSON schema the final answer must follow; `None` for plain text."""
    media: tuple[Modality, ...]
    """What the answer must carry besides text: a spoken clip, an image."""
    iteration: int
    """0 for the first provider call of a run, incremented after each round of tool calls."""
    provider: Provider
    """The provider this step will call; middleware may also use it directly, e.g. to summarize."""
    thread: Conversation
    """The conversation this request belongs to."""
    deferred: tuple[Tool, ...] = ()
    """Tools declared to the provider but kept out of the model's context until a `ToolResult.tools` in
    `messages` names them, which is what `lazy()`'s search tool does. A provider with deferred loading
    (the Messages API) sends them all and lets the API expand the loaded ones in place; one without sends
    only `loaded`. Either way the model can call a loaded tool."""

    @property
    def loaded(self) -> tuple[Tool, ...]:
        """The deferred tools a result in the conversation has loaded, in catalog order."""
        names = {name for m in self.messages if isinstance(m, ToolResult) for name in m.tools}
        return tuple(t for t in self.deferred if t.name in names)
