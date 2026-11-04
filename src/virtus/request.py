from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from virtus.message import Message

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
