from dataclasses import dataclass


@dataclass(frozen=True)
class UserMessage:
    content: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    """Raw JSON, exactly as produced by the model; validated by the tool."""


@dataclass(frozen=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant text, streamed before its complete `AssistantMessage`."""

    content: str


type Message = UserMessage | AssistantMessage | ToolResult
type Event = TextDelta | Message
