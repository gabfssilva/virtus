from typing import TYPE_CHECKING

from virtus.agent import Agent, agent
from virtus.capability import Capabilities, context, on_input, on_message
from virtus.compaction import compaction
from virtus.journal import Journal, JsonJournal, MemoryJournal
from virtus.knowledge import Document, Store, knowledge, memory
from virtus.live import Interrupted, Level, Live
from virtus.message import (
    AssistantMessage,
    Audio,
    AudioDelta,
    AudioType,
    Delta,
    Event,
    Image,
    ImageType,
    MediaType,
    Message,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    UserMessage,
)
from virtus.middleware import Capability, Middleware, Next
from virtus.observe import observe
from virtus.output import Snapshot
from virtus.provider import LiveProvider, Provider, Session
from virtus.request import Request
from virtus.thread import Conversation, Thread
from virtus.tool import Tool, tool
from virtus.vault import secrets
from virtus.workspace import workspace

if TYPE_CHECKING:
    from virtus.servers import Http, Stdio, mcp

__all__ = [
    "Agent",
    "AssistantMessage",
    "Audio",
    "AudioDelta",
    "AudioType",
    "Capabilities",
    "Capability",
    "Conversation",
    "Delta",
    "Document",
    "Event",
    "Http",
    "Image",
    "ImageType",
    "Interrupted",
    "Journal",
    "JsonJournal",
    "Level",
    "Live",
    "LiveProvider",
    "MediaType",
    "MemoryJournal",
    "Message",
    "Middleware",
    "Next",
    "Provider",
    "Request",
    "Session",
    "Snapshot",
    "Stdio",
    "Store",
    "TextDelta",
    "Thread",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolResult",
    "UserMessage",
    "agent",
    "compaction",
    "context",
    "knowledge",
    "mcp",
    "memory",
    "observe",
    "on_input",
    "on_message",
    "secrets",
    "tool",
    "workspace",
]


def __getattr__(name: str) -> object:
    """`mcp`, `Stdio` and `Http` load on demand, so the `mcp` extra is only needed by those who use it."""
    if name in ("Http", "Stdio", "mcp"):
        from virtus import servers

        return getattr(servers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
