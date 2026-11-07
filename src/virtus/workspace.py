"""The agent's files: a directory it can read, write, edit, list and search, and nothing outside it."""

import re
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

from virtus.message import Event
from virtus.middleware import Middleware, Next
from virtus.request import Request
from virtus.tool import tool


def workspace(path: str | Path, *, name: str = "workspace", max_chars: int = 20_000, max_hits: int = 50) -> Middleware:
    """Give the agent the files under `path` through `{name}_read`, `_write`, `_edit`, `_list` and `_search`.

    Paths in tool calls are relative to `path`; one that resolves outside it is refused. Hidden entries
    are skipped by `list` and `search`.
    """
    root = Path(path).resolve()

    def locate(relative: str) -> Path:
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise PermissionError(f"{relative!r} is outside the {name}")
        return target

    def read(path: str) -> str:
        """Read a text file."""
        text = locate(path).read_text()
        return text if len(text) <= max_chars else f"{text[:max_chars]}\n… [truncated; {len(text)} chars in total]"

    def write(path: str, content: str) -> str:
        """Create or overwrite a file, creating its folders as needed."""
        target = locate(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def edit(path: str, old: str, new: str) -> str:
        """Replace `old`, which must occur exactly once in the file, with `new`."""
        target = locate(path)
        text = target.read_text()
        match text.count(old):
            case 0:
                raise ValueError(f"{old!r} not found in {path}")
            case 1:
                target.write_text(text.replace(old, new))
                return f"edited {path}"
            case times:
                raise ValueError(f"{old!r} occurs {times} times in {path}; include more context")

    def listing(path: str = ".") -> str:
        """List a folder; folders end with a slash."""
        entries = sorted(locate(path).iterdir(), key=lambda entry: (not entry.is_dir(), entry.name))
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries if not e.name.startswith(".")]
        return "\n".join(lines) or "(empty)"

    def search(pattern: str, path: str = ".") -> str:
        """Search the text files under a folder with a regular expression; returns `file:line: text`."""
        regex = re.compile(pattern)
        hits: list[str] = []
        for file in sorted(locate(path).rglob("*")):
            relative = file.relative_to(root)
            if not file.is_file() or any(part.startswith(".") for part in relative.parts):
                continue
            try:
                lines = file.read_text().splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if regex.search(line):
                    hits.append(f"{relative}:{number}: {line.strip()}")
                if len(hits) == max_hits:
                    return "\n".join(hits) + "\n… [more hits; narrow the pattern]"
        return "\n".join(hits) or "No matches."

    tools = (
        tool(read, name=f"{name}_read"),
        tool(write, name=f"{name}_write"),
        tool(edit, name=f"{name}_edit"),
        tool(listing, name=f"{name}_list"),
        tool(search, name=f"{name}_search"),
    )
    note = f"Your {name} is a folder of files you reach with the {name}_* tools; paths are relative to its root."

    async def provide(request: Request, next: Next) -> AsyncIterator[Event]:
        system = "\n\n".join(part for part in (request.system, note) if part)
        async for event in next(replace(request, system=system, tools=(*request.tools, *tools))):
            yield event

    return provide
