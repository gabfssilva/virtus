"""Tools served over the Model Context Protocol.

`async with mcp(fs=...) as tools:` connects to each named server for the duration of the block and
yields one capability offering every tool they serve, named `fs__read_file` after the server. The
tools keep the schema the server declared and send the arguments through as the model wrote them;
the server validates. Needs the `mcp` extra.
"""

import json
import shlex
from base64 import b64decode
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx2
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams
from pydantic import BaseModel
from pydantic_core import to_json

from virtus.message import Audio, Event, Image, ToolCall, ToolResult
from virtus.middleware import Capability, Middleware, Next
from virtus.request import Request
from virtus.tool import Tool


@dataclass(frozen=True)
class Stdio:
    """A server run as a subprocess and talked to over its stdin and stdout.

    The process gets `env` on top of a minimal environment (`HOME`, `PATH`, `USER` and the like); the
    shell's other variables do not reach it unless listed here.
    """

    command: str
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None


@dataclass(frozen=True)
class Http:
    """A server reached over streamable HTTP; `headers` go on every request.

    `auth` is any `httpx2.Auth`, such as the SDK's `OAuthClientProvider` for a server behind OAuth.
    `timeout` is in seconds, `read_timeout` longer by default because a reply may be a long-lived stream.
    """

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    auth: httpx2.Auth | None = None
    timeout: float = 30
    read_timeout: float = 300


@asynccontextmanager
async def mcp(**servers: Stdio | Http | str) -> AsyncIterator["Served"]:
    """Connect to these servers for the block and offer their tools as one capability.

    Each keyword names a server and prefixes its tools: `mcp(fs="npx -y @modelcontextprotocol/server-filesystem .")`
    offers `fs__read_file`. A string is a command line, or a URL when it starts with `http://` or
    `https://`; `Stdio(...)` and `Http(...)` add environment variables and headers. A server that
    cannot be reached fails the block before any agent runs.
    """
    async with AsyncExitStack() as stack:

        async def connect(name: str, server: Stdio | Http) -> list[Tool]:
            match server:
                case Stdio(command, env, cwd):
                    program, *args = shlex.split(command)
                    parameters = StdioServerParameters(command=program, args=args, env=dict(env), cwd=cwd)
                    read, write = await stack.enter_async_context(stdio_client(parameters))
                case Http(url, headers, auth, seconds, read_seconds):
                    timeout = httpx2.Timeout(seconds, read=read_seconds)
                    http = httpx2.AsyncClient(headers=dict(headers), timeout=timeout, auth=auth)
                    client = await stack.enter_async_context(http)
                    read, write = await stack.enter_async_context(streamable_http_client(url, http_client=client))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return [Remote.of(name, session, tool) for tool in await _all_tools(session)]

        tools = [tool for name, server in servers.items() for tool in await connect(name, _spec(server))]
        yield Served(tuple(tools))


@dataclass(frozen=True)
class Served(Capability):
    """The tools of connected servers, added to every step."""

    tools: tuple[Tool, ...]

    def middleware(self) -> Sequence[Middleware]:
        async def provide(request: Request, next: Next) -> AsyncIterator[Event]:
            async for event in next(replace(request, tools=(*request.tools, *self.tools))):
                yield event

        return (provide,)


@dataclass(frozen=True)
class Remote(Tool):
    """A tool a server runs: the server's schema, the model's arguments passed through unparsed."""

    session: ClientSession
    remote: str
    input_schema: Mapping[str, object]

    @classmethod
    def of(cls, prefix: str, session: ClientSession, tool: types.Tool) -> "Remote":
        name = f"{prefix}__{tool.name}"
        return cls(name, tool.description or "", BaseModel, session.call_tool, session, tool.name, tool.input_schema)

    @property
    def schema(self) -> Mapping[str, object]:
        return self.input_schema

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            result = await self.session.call_tool(self.remote, json.loads(call.arguments or "{}"))
        except Exception as error:
            return ToolResult(call.id, call.name, f"{type(error).__name__}: {error}", is_error=True)
        if not isinstance(result, types.CallToolResult):
            return ToolResult(call.id, call.name, f"unsupported reply: {type(result).__name__}", is_error=True)

        texts: list[str] = []
        images: list[Image] = []
        audio: list[Audio] = []
        for block in result.content:
            match block:
                case types.TextContent(text=text):
                    texts.append(text)
                case types.ImageContent(data=data, mime_type=mime) | types.AudioContent(data=data, mime_type=mime):
                    match _media(data, mime):
                        case Image() as image:
                            images.append(image)
                            texts.append("[image]")
                        case Audio() as clip:
                            audio.append(clip)
                            texts.append("[audio]")
                        case None:
                            texts.append(f"[{mime}, {len(b64decode(data))} bytes]")
                case types.ResourceLink(uri=uri):
                    texts.append(str(uri))
                case types.EmbeddedResource(resource=types.TextResourceContents(text=text)):
                    texts.append(text)
                case types.EmbeddedResource(resource=resource):
                    texts.append(f"[{resource.mime_type or 'resource'}, {resource.uri}]")
        if not texts and result.structured_content is not None:
            texts.append(to_json(result.structured_content).decode())
        content = "\n".join(texts)
        return ToolResult(call.id, call.name, content, bool(result.is_error), tuple(images), tuple(audio))


async def _all_tools(session: ClientSession) -> list[types.Tool]:
    """Every page of the server's tool list."""
    tools: list[types.Tool] = []
    cursor: str | None = None
    while True:
        page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools


def _spec(server: Stdio | Http | str) -> Stdio | Http:
    match server:
        case str() if server.startswith(("http://", "https://")):
            return Http(server)
        case str():
            return Stdio(server)
        case _:
            return server


def _media(data: str, mime: str) -> Image | Audio | None:
    raw = b64decode(data)
    match mime:
        case "image/jpeg" | "image/png" | "image/gif" | "image/webp" as kind:
            return Image(raw, kind)
        case "audio/wav" | "audio/mpeg" | "audio/flac" | "audio/ogg" | "audio/aac" as kind:
            return Audio(raw, kind)
        case _:
            return None
