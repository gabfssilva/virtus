"""Where threads keep their history, so a conversation outlives the process and an interrupted run resumes.

A thread with a journal writes its whole history after every completed step, the same moment the
in-memory history changes. The history is all the framework needs to continue: tool calls never sit
without their results in it, and rewrites made by middleware are already applied. State that lives
elsewhere (a `Capabilities` instance, a kernel) is not covered; what needs to survive goes to disk by
its owner, or is rebuilt from the history.
"""

import shutil
from base64 import b64decode, b64encode
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, Field, TypeAdapter

from virtus.message import AssistantMessage, Image, MediaType, Message, ToolCall, ToolResult, UserMessage


@runtime_checkable
class Journal(Protocol):
    """`path` is `(id,)` for a root thread; a thread opened by delegation appends the agent name and the
    conversation name to its parent's path. `delete` removes the thread and every thread under it.
    """

    def read(self, path: Sequence[str]) -> tuple[Message, ...] | None: ...

    def write(self, path: Sequence[str], history: Sequence[Message]) -> None: ...

    def delete(self, path: Sequence[str]) -> None: ...


class JsonJournal(Journal):
    """One JSON file per thread under `directory`; children live in a folder named after their parent."""

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

    def _file(self, path: Sequence[str]) -> Path:
        """Each segment becomes one file-system name; percent-encoding keeps `/` and `..` out of it."""
        segments = [quote(segment, safe="").replace(".", "%2E") for segment in path]
        return self.directory.joinpath(*segments).with_suffix(".json")


class _Image(BaseModel):
    media_type: MediaType
    data: str

    @classmethod
    def of(cls, image: Image) -> "_Image":
        return cls(media_type=image.media_type, data=b64encode(image.data).decode())

    def image(self) -> Image:
        return Image(b64decode(self.data), self.media_type)


class _User(BaseModel):
    role: Literal["user"] = "user"
    content: str
    images: list[_Image] = []


class _Call(BaseModel):
    id: str
    name: str
    arguments: str


class _Assistant(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[_Call] = []


class _Tool(BaseModel):
    role: Literal["tool"] = "tool"
    call_id: str
    name: str
    content: str
    is_error: bool = False
    images: list[_Image] = []


type _Entry = Annotated[_User | _Assistant | _Tool, Field(discriminator="role")]


class _Record(BaseModel):
    version: Literal[1] = 1
    messages: list[_Entry]


_RECORD = TypeAdapter(_Record)


def _entry(message: Message) -> _User | _Assistant | _Tool:
    match message:
        case UserMessage(content=content, images=images):
            return _User(content=content, images=[_Image.of(i) for i in images])
        case AssistantMessage(content=content, tool_calls=calls):
            tool_calls = [_Call(id=c.id, name=c.name, arguments=c.arguments) for c in calls]
            return _Assistant(content=content, tool_calls=tool_calls)
        case ToolResult(call_id=call_id, name=name, content=content, is_error=is_error, images=images):
            encoded = [_Image.of(i) for i in images]
            return _Tool(call_id=call_id, name=name, content=content, is_error=is_error, images=encoded)


def _message(entry: _User | _Assistant | _Tool) -> Message:
    match entry:
        case _User(content=content, images=images):
            return UserMessage(content, tuple(i.image() for i in images))
        case _Assistant(content=content, tool_calls=calls):
            return AssistantMessage(content, tuple(ToolCall(c.id, c.name, c.arguments) for c in calls))
        case _Tool(call_id=call_id, name=name, content=content, is_error=is_error, images=images):
            return ToolResult(call_id, name, content, is_error, tuple(i.image() for i in images))
