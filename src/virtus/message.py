import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type MediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: MediaType

    @classmethod
    def read(cls, path: str | Path) -> "Image":
        """Load an image file, taking the media type from its extension."""
        match mimetypes.guess_type(str(path))[0]:
            case "image/jpeg" | "image/png" | "image/gif" | "image/webp" as media_type:
                return cls(Path(path).read_bytes(), media_type)
            case other:
                raise ValueError(f"{path}: unsupported image type {other!r}")

    @staticmethod
    def found_in(value: object) -> tuple["Image", ...]:
        """The images a value carries: itself, or the items of a sequence (not a string) that are images."""
        if isinstance(value, Image):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return tuple(v for v in value if isinstance(v, Image))
        return ()


@dataclass(frozen=True)
class UserMessage:
    content: str
    images: tuple[Image, ...] = ()


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
    images: tuple[Image, ...] = ()


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
