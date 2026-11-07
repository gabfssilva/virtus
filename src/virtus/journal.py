"""Where threads keep their history: the only state a conversation has, so it outlives the process and resumes.

A thread holds no messages of its own; it reads its history from the journal before each run and writes
it back after every completed step. `MemoryJournal` is the default and lives as long as the agent;
`JsonJournal` puts the same thing on disk. The history is all the framework needs to continue: tool calls
never sit without their results in it, and rewrites made by middleware are already applied. State that
lives elsewhere (a `Capabilities` instance, a kernel) is not covered; what needs to survive goes to disk
by its owner, or is rebuilt from the history.
"""

import shutil
from base64 import b64decode, b64encode
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable
from urllib.parse import quote, unquote

from pydantic import BaseModel, Field, TypeAdapter

from virtus.message import (
    AssistantMessage,
    Audio,
    AudioType,
    Image,
    ImageType,
    Message,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
)


@runtime_checkable
class Journal(Protocol):
    """`path` is `(agent name, id)` for a root thread; a thread opened by delegation appends the agent name
    and the conversation name to its parent's path. `delete` removes the thread and every thread under it;
    `children` lists the (agent name, conversation name) pairs directly under a thread.
    """

    def read(self, path: Sequence[str]) -> tuple[Message, ...] | None: ...

    def write(self, path: Sequence[str], history: Sequence[Message]) -> None: ...

    def delete(self, path: Sequence[str]) -> None: ...

    def children(self, path: Sequence[str]) -> Sequence[tuple[str, str]]: ...


class MemoryJournal(Journal):
    """Threads kept in a dict, for as long as the journal is referenced."""

    def __init__(self) -> None:
        self._threads: dict[tuple[str, ...], tuple[Message, ...]] = {}

    def read(self, path: Sequence[str]) -> tuple[Message, ...] | None:
        return self._threads.get(tuple(path))

    def write(self, path: Sequence[str], history: Sequence[Message]) -> None:
        self._threads[tuple(path)] = tuple(history)

    def delete(self, path: Sequence[str]) -> None:
        prefix = tuple(path)
        for key in [k for k in self._threads if k[: len(prefix)] == prefix]:
            del self._threads[key]

    def children(self, path: Sequence[str]) -> tuple[tuple[str, str], ...]:
        prefix = tuple(path)
        return tuple((k[-2], k[-1]) for k in self._threads if len(k) == len(prefix) + 2 and k[: len(prefix)] == prefix)


class JsonJournal(Journal):
    """One JSON file per thread, `directory/<agent>/<id>.json`; children live in a folder named after their parent."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def read(self, path: Sequence[str]) -> tuple[Message, ...] | None:
        file = self._file(path)
        if not file.exists():
            return None
        return tuple(map(_message, _RECORD.validate_json(file.read_bytes()).messages))

    def write(self, path: Sequence[str], history: Sequence[Message]) -> None:
        file = self._file(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        record = _Record(messages=[_entry(m) for m in history])
        temporary = file.with_suffix(".json.tmp")
        temporary.write_bytes(record.model_dump_json(indent=1).encode())
        temporary.replace(file)

    def delete(self, path: Sequence[str]) -> None:
        self._file(path).unlink(missing_ok=True)
        shutil.rmtree(self._file(path).with_suffix(""), ignore_errors=True)

    def children(self, path: Sequence[str]) -> tuple[tuple[str, str], ...]:
        folder = self._file(path).with_suffix("")
        if not folder.is_dir():
            return ()
        return tuple(
            (unquote(agent.name), unquote(file.stem))
            for agent in sorted(folder.iterdir())
            if agent.is_dir()
            for file in sorted(agent.glob("*.json"))
        )

    def _file(self, path: Sequence[str]) -> Path:
        """Each segment becomes one file-system name; percent-encoding keeps `/` and `..` out of it."""
        segments = [quote(segment, safe="").replace(".", "%2E") for segment in path]
        return self.directory.joinpath(*segments).with_suffix(".json")


class _Image(BaseModel):
    media_type: ImageType
    data: str

    @classmethod
    def of(cls, image: Image) -> "_Image":
        return cls(media_type=image.media_type, data=b64encode(image.data).decode())

    def image(self) -> Image:
        return Image(b64decode(self.data), self.media_type)


class _Audio(BaseModel):
    media_type: AudioType
    data: str

    @classmethod
    def of(cls, audio: Audio) -> "_Audio":
        return cls(media_type=audio.media_type, data=b64encode(audio.data).decode())

    def audio(self) -> Audio:
        return Audio(b64decode(self.data), self.media_type)


class _User(BaseModel):
    role: Literal["user"] = "user"
    content: str
    images: list[_Image] = []
    audio: list[_Audio] = []


class _Call(BaseModel):
    id: str
    name: str
    arguments: str


class _Usage(BaseModel):
    input: int
    output: int


class _Assistant(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[_Call] = []
    images: list[_Image] = []
    audio: _Audio | None = None
    usage: _Usage | None = None


class _Tool(BaseModel):
    role: Literal["tool"] = "tool"
    call_id: str
    name: str
    content: str
    is_error: bool = False
    images: list[_Image] = []
    audio: list[_Audio] = []


type _Entry = Annotated[_User | _Assistant | _Tool, Field(discriminator="role")]


class _Record(BaseModel):
    version: Literal[1] = 1
    messages: list[_Entry]


_RECORD = TypeAdapter(_Record)


def _entry(message: Message) -> _User | _Assistant | _Tool:
    match message:
        case UserMessage(content=content, images=images, audio=audio):
            return _User(content=content, images=_images(images), audio=_clips(audio))
        case AssistantMessage(content=content, tool_calls=calls, images=images, audio=audio, usage=usage):
            tool_calls = [_Call(id=c.id, name=c.name, arguments=c.arguments) for c in calls]
            spoken = _Audio.of(audio) if audio else None
            used = _Usage(input=usage.input, output=usage.output) if usage else None
            return _Assistant(content=content, tool_calls=tool_calls, images=_images(images), audio=spoken, usage=used)
        case ToolResult(call_id=call_id, name=name, content=content, is_error=is_error, images=images, audio=audio):
            entry = _Tool(call_id=call_id, name=name, content=content, is_error=is_error)
            return entry.model_copy(update={"images": _images(images), "audio": _clips(audio)})


def _message(entry: _User | _Assistant | _Tool) -> Message:
    match entry:
        case _User(content=content, images=images, audio=audio):
            return UserMessage(content, tuple(i.image() for i in images), tuple(a.audio() for a in audio))
        case _Assistant(content=content, tool_calls=calls, images=images, audio=audio, usage=usage):
            tool_calls = tuple(ToolCall(c.id, c.name, c.arguments) for c in calls)
            spoken = audio.audio() if audio else None
            used = Usage(usage.input, usage.output) if usage else None
            return AssistantMessage(content, tool_calls, tuple(i.image() for i in images), spoken, used)
        case _Tool(call_id=call_id, name=name, content=content, is_error=is_error, images=images, audio=audio):
            attached = tuple(i.image() for i in images), tuple(a.audio() for a in audio)
            return ToolResult(call_id, name, content, is_error, *attached)


def _images(images: Sequence[Image]) -> list[_Image]:
    return [_Image.of(i) for i in images]


def _clips(audio: Sequence[Audio]) -> list[_Audio]:
    return [_Audio.of(a) for a in audio]
