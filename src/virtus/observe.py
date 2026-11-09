"""Each run drawn as a tree while it happens; see `observe`."""

import json
import sys
from collections.abc import AsyncIterator
from time import perf_counter
from typing import TextIO

from virtus.message import (
    AssistantMessage,
    AudioDelta,
    Event,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    UserMessage,
)
from virtus.middleware import Middleware, Next
from virtus.request import Request

WIDTH = 100


def observe(out: TextIO | None = None, *, color: bool | None = None, width: int = WIDTH) -> Middleware:
    """Show every run as a tree, as it happens: each provider call with its message count and tools, the
    tool calls and what they returned, the answer streamed, and how long each part took.

    Writes to stderr unless `out` is given, in color when that is a terminal; a prompt, a tool's arguments
    or a tool's result is cut to `width` characters. Goes anywhere in
    `capabilities=`: last, it shows what the provider receives, tools included; first, what the caller sees.
    """
    console = _Console(out or sys.stderr, color, width)
    runs: dict[str, _Run] = {}

    async def observe(request: Request, next: Next) -> AsyncIterator[Event]:
        key = "/".join(request.thread.path)
        if request.iteration == 0 or key not in runs:
            runs[key] = _Run(console, request)
        run = runs[key]
        run.step(request)
        try:
            async for event in next(request):
                run.saw(event)
                yield event
        except Exception as error:
            run.failed(error)
            raise
        finally:
            if run.done:
                runs.pop(key, None)

    return observe


class _Console:
    def __init__(self, out: TextIO, color: bool | None, width: int) -> None:
        self.width = width
        self.out = out
        self.color = out.isatty() if color is None else color

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def dim(self, text: str) -> str:
        return self._paint(2, text)

    def cyan(self, text: str) -> str:
        return self._paint(36, text)

    def green(self, text: str) -> str:
        return self._paint(32, text)

    def red(self, text: str) -> str:
        return self._paint(31, text)

    def _paint(self, code: int, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color and text else text


class _Run:
    """One run of one thread: from the user's message to the answer, over as many steps as it takes."""

    def __init__(self, console: _Console, request: Request) -> None:
        self.console = console
        self.started = perf_counter()
        self.done = False
        self.step_started = self.started
        self.first_token: float | None = None
        self.typing = False
        self.heard = 0
        self.tools_started = self.started
        asked = request.messages[-1] if request.messages else None
        prompt = f" · {_clip(asked.content, self.console.width)}" if isinstance(asked, UserMessage) else ""
        console.line(f"▶ {'/'.join(request.thread.path)}{console.dim(prompt)}")

    def step(self, request: Request) -> None:
        self.step_started = perf_counter()
        self.first_token = None
        self.typing = False
        self.heard = 0
        tools = f" · tools: {', '.join(t.name for t in request.tools)}" if request.tools else ""
        if request.deferred:
            tools += f" · {len(request.loaded)}/{len(request.deferred)} deferred tools loaded"
        count = _plural(len(request.messages), "msg")
        self.console.line(f"├─ {request.provider!r} · {count}{self.console.dim(tools)}")

    def saw(self, event: Event) -> None:
        console = self.console
        match event:
            case TextDelta(content=text):
                self._token()
                self._type(text)
            case AudioDelta(data=data):
                self._token()
                self.heard += len(data)
            case ToolCallDelta():
                self._token()
            case AssistantMessage(content=content, tool_calls=calls, usage=usage):
                self._end_typing()
                elapsed = f"{perf_counter() - self.step_started:.1f}s"
                first = f" · first token {self.first_token:.1f}s" if self.first_token is not None else ""
                first += f" · {usage.input} in, {usage.output} out" if usage else ""
                if calls:
                    console.line(console.dim(f"│  {elapsed}{first}"))
                    for call in calls:
                        console.line(f"│  {console.cyan(f'⚙ {call.name}')}({_arguments(call, console.width)})")
                    self.tools_started = perf_counter()
                else:
                    size = f"{len(content)} chars" if content else ""
                    if self.heard:
                        size = f"{size} · ♪ {self.heard / 48_000:.1f}s".lstrip(" ·")  # 24 kHz, 16-bit mono
                    total = f"{perf_counter() - self.started:.1f}s total"
                    detail = " · ".join(part for part in (size, f"{elapsed}{first}", total) if part)
                    console.line(f"└─ {console.green('✓')} {console.dim(detail)}")
                    self.done = True
            case ToolResult(name=name, content=content, is_error=error):
                took = console.dim(f"· {perf_counter() - self.tools_started:.1f}s")
                mark = console.red(f"✗ {name}") if error else console.dim("↳")
                console.line(f"│  {mark} {_clip(content, console.width)} {took}")
            case _:
                pass

    def failed(self, error: Exception) -> None:
        self._end_typing()
        self.console.line(f"└─ {self.console.red('✗')} {type(error).__name__}: {_clip(str(error), self.console.width)}")
        self.done = True

    def _token(self) -> None:
        if self.first_token is None:
            self.first_token = perf_counter() - self.step_started

    def _type(self, text: str) -> None:
        if not self.typing:
            self.console.write("│  ")
            self.typing = True
        self.console.write(text.replace("\n", "\n│  "))

    def _end_typing(self) -> None:
        if self.typing:
            self.console.write("\n")
            self.typing = False


def _arguments(call: ToolCall, width: int) -> str:
    try:
        arguments = json.loads(call.arguments or "{}")
    except ValueError:
        return _clip(call.arguments, width)
    if not isinstance(arguments, dict):
        return _clip(call.arguments, width)
    pairs = (f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in arguments.items())
    return _clip(", ".join(pairs), width)


def _clip(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
