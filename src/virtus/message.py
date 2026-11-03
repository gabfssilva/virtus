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


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of a tool call's arguments, streamed before the `AssistantMessage` carrying the whole call.

    The id and name are known from the first fragment, whose `arguments` may be empty.
    """

    id: str
    name: str
    arguments: str


type Delta = TextDelta | ToolCallDelta
type Message = UserMessage | AssistantMessage | ToolResult
type Event = Delta | Message
