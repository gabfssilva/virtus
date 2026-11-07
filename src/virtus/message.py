import io
import mimetypes
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type ImageType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
type AudioType = Literal["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/aac"]
type MediaType = ImageType | AudioType


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: ImageType

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
        return attached(Image, value)


@dataclass(frozen=True)
class Audio:
    data: bytes
    media_type: AudioType

    @classmethod
    def read(cls, path: str | Path) -> "Audio":
        """Load an audio file, taking the media type from its extension."""
        extensions: dict[str, AudioType] = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
            ".aac": "audio/aac",
        }
        if media_type := extensions.get(Path(path).suffix.lower()):
            return cls(Path(path).read_bytes(), media_type)
        raise ValueError(f"{path}: unsupported audio type")

    @classmethod
    def pcm16(cls, data: bytes, rate: int = 24_000) -> "Audio":
        """Raw 16-bit mono PCM as a WAV clip, the form speech models stream in and out."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(rate)
            file.writeframes(data)
        return cls(buffer.getvalue(), "audio/wav")

    @staticmethod
    def found_in(value: object) -> tuple["Audio", ...]:
        """The audio clips a value carries: itself, or the items of a sequence (not a string) that are audio."""
        return attached(Audio, value)


def attached[T](kind: type[T], value: object) -> tuple[T, ...]:
    if isinstance(value, kind):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(v for v in value if isinstance(v, kind))
    return ()


def placeholder(value: object) -> str | None:
    """How a media value reads in text: `[image]` or `[audio]`; `None` for anything else."""
    if Image.found_in(value):
        return "[image]"
    if Audio.found_in(value):
        return "[audio]"
    return None


@dataclass(frozen=True)
class UserMessage:
    content: str
    images: tuple[Image, ...] = ()
    audio: tuple[Audio, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    """Raw JSON, exactly as produced by the model; validated by the tool."""


@dataclass(frozen=True)
class Usage:
    """Tokens one provider call consumed: the prompt as `input`, the reply as `output`."""

    input: int = 0
    output: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input + other.input, self.output + other.output)


@dataclass(frozen=True)
class AssistantMessage:
    content: str = ""
    """The text, or the transcript when the answer is spoken."""
    tool_calls: tuple[ToolCall, ...] = ()
    images: tuple[Image, ...] = ()
    audio: Audio | None = None
    usage: Usage | None = None
    """What the call that produced this message consumed, when the provider reports it."""


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    images: tuple[Image, ...] = ()
    audio: tuple[Audio, ...] = ()


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant text, streamed before its complete `AssistantMessage`."""

    content: str


@dataclass(frozen=True)
class AudioDelta:
    """A fragment of the spoken answer, streamed before the `AssistantMessage` carrying the whole clip."""

    data: bytes


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of a tool call's arguments, streamed before the `AssistantMessage` carrying the whole call.

    The id and name are known from the first fragment, whose `arguments` may be empty.
    """

    id: str
    name: str
    arguments: str


type Delta = TextDelta | AudioDelta | ToolCallDelta
type Message = UserMessage | AssistantMessage | ToolResult
type Event = Delta | Message
