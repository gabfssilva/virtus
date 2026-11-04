"""Documents on disk the agent can consult, retrieved in the cheapest way their volume allows.

While everything fits the token budget, it all goes into the system prompt. Past that, only an
index of ids and titles goes in and the agent reads entries on demand. Past what the index
allows, the agent searches first (BM25, no dependencies) and then reads.

`knowledge(...)` is read-only material the developer provides; `memory(...)` is the same store
with tools for the agent to write and delete entries, so it persists across conversations.
"""

import math
import re
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from virtus.message import Event
from virtus.middleware import Middleware, Next
from virtus.request import Request
from virtus.tool import Tool, tool


@dataclass(frozen=True)
class Document:
    id: str
    """Path relative to the root it was found under, e.g. `guides/setup.md`."""
    title: str
    content: str
    path: Path


class Store:
    """Markdown and text files under `roots`, read again on every access so edits show up at once.

    A `.md` file may start with a frontmatter block of `key: value` lines between `---` fences;
    its `title` names the document. Otherwise the first heading does, and failing that the file name.
    """

    def __init__(self, *roots: Path, suffixes: Sequence[str] = (".md", ".txt")) -> None:
        self.roots = roots
        self.suffixes = tuple(suffixes)

    def documents(self) -> tuple[Document, ...]:
        def under(root: Path) -> Iterator[Document]:
            if root.is_file():
                yield load(root, root.name)
            elif root.is_dir():
                for file in sorted(root.rglob("*")):
                    if file.suffix in self.suffixes:
                        try:
                            yield load(file, str(file.relative_to(root)))
                        except (FileNotFoundError, IsADirectoryError):
                            continue  # deleted, or a directory named like a file, while listing

        return tuple(d for root in self.roots for d in under(root))

    def read(self, id: str) -> Document | None:
        return next((d for d in self.documents() if d.id == id), None)

    def search(self, query: str, limit: int = 5) -> list[Document]:
        """Rank documents by BM25 (k1=1.5, b=0.75) over title and content; only positive scores count."""
        documents = self.documents()
        bags = [Counter(_tokens(f"{d.title} {d.content}")) for d in documents]
        total = len(documents)
        average = sum(sum(bag.values()) for bag in bags) / total if total else 0.0

        def idf(term: str) -> float:
            containing = sum(term in bag for bag in bags)
            return math.log(1 + (total - containing + 0.5) / (containing + 0.5))

        def score(bag: Counter[str]) -> float:
            length = sum(bag.values())
            return sum(
                idf(term) * bag[term] * 2.5 / (bag[term] + 1.5 * (0.25 + 0.75 * length / average))
                for term in set(_tokens(query))
                if term in bag
            )

        ranked = sorted(zip(map(score, bags), documents, strict=True), key=lambda pair: -pair[0])
        return [document for value, document in ranked[:limit] if value > 0]


def knowledge(*paths: str | Path, name: str = "knowledge", max_tokens: int = 4_000) -> Middleware:
    """Let the agent consult `.md` and `.txt` files, or whole directories of them, read-only.

    `name` prefixes the tools and the section in the system prompt, so two knowledge bases can
    coexist. `max_tokens` (estimated as four characters each) is how much of the prompt the
    content, or failing that the index, may take before the agent has to read or search instead.
    """
    return _consult(
        Store(*map(Path, paths)),
        name,
        max_tokens,
        intro="Reference material you can rely on.",
        manage=(),
    )


def memory(path: str | Path, *, name: str = "memory", max_tokens: int = 4_000) -> Middleware:
    """Give the agent a directory of markdown notes it writes and reads back in later conversations.

    Each note is a file named after its title, with the title and creation time in frontmatter.
    Notes are plain files, so they can be read and edited by hand.
    """
    root = Path(path)
    store = Store(root)

    def remember(title: str, content: str) -> str:
        """Save something worth knowing in later conversations. The same title overwrites the note."""
        root.mkdir(parents=True, exist_ok=True)
        file = root / f"{_slug(title)}.md"
        created = datetime.now(UTC).isoformat(timespec="seconds")
        file.write_text(f"---\ntitle: {title}\ncreated: {created}\n---\n\n{content.strip()}\n")
        return f"saved as {file.name}"

    def forget(id: str) -> str:
        """Delete a note by id."""
        if (document := store.read(id)) is None:
            raise FileNotFoundError(f"no note {id!r}")
        document.path.unlink()
        return f"deleted {id}"

    return _consult(
        store,
        name,
        max_tokens,
        intro=f"Notes you kept from earlier conversations. Save new ones with {name}_remember.",
        manage=(tool(remember, name=f"{name}_remember"), tool(forget, name=f"{name}_forget")),
    )


def load(file: Path, id: str) -> Document:
    """A heading that names a markdown document is not repeated in its content."""
    if file.suffix != ".md":
        return Document(id, file.stem, file.read_text().strip(), file)

    frontmatter, body = _split(file.read_text())
    if title := frontmatter.get("title"):
        return Document(id, title, body.strip(), file)
    if heading := re.search(r"^#\s+(.+)$", body, re.MULTILINE):
        return Document(id, heading.group(1).strip(), (body[: heading.start()] + body[heading.end() :]).strip(), file)
    return Document(id, file.stem, body.strip(), file)


def _consult(store: Store, name: str, max_tokens: int, *, intro: str, manage: tuple[Tool, ...]) -> Middleware:
    def read(id: str) -> str:
        if (document := store.read(id)) is None:
            raise FileNotFoundError(f"no entry {id!r}")
        return f"# {document.title}\n\n{document.content}"

    def search(query: str) -> str:
        hits = store.search(query)
        return "\n".join(f"- {d.id}: {d.title}\n  {_snippet(d.content)}" for d in hits) or "No matches."

    reader = tool(read, name=f"{name}_read", description=f"Read one entry of {name} by id.")
    searcher = tool(search, name=f"{name}_search", description=f"Search {name} by keywords; returns ids to read.")

    async def consult(request: Request, next: Next) -> AsyncIterator[Event]:
        documents = store.documents()
        full = "\n\n".join(f"### {d.title} ({d.id})\n{d.content}" for d in documents)
        index = "\n".join(f"- {d.id}: {d.title}" for d in documents)

        if len(full) // 4 <= max_tokens:
            body, tools = full or "(nothing yet)", manage
        elif len(index) // 4 <= max_tokens:
            body, tools = f"{index}\n\nRead an entry with {name}_read.", (reader, *manage)
        else:
            body = f"{len(documents)} entries. Find the relevant ones with {name}_search, then {name}_read them."
            tools = (searcher, reader, *manage)

        section = f"## {name}\n\n{intro}\n\n{body}"
        system = "\n\n".join(part for part in (request.system, section) if part)

        async for event in next(replace(request, system=system, tools=(*request.tools, *tools))):
            yield event

    return consult


_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def _split(text: str) -> tuple[dict[str, str], str]:
    if (match := _FRONTMATTER.match(text)) is None:
        return {}, text
    pairs = (line.partition(":") for line in match.group(1).splitlines() if ":" in line)
    return {key.strip(): value.strip().strip("'\"") for key, _, value in pairs}, text[match.end() :]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "note"


def _snippet(content: str, length: int = 160) -> str:
    collapsed = " ".join(content.split())
    return collapsed if len(collapsed) <= length else f"{collapsed[:length].rstrip()}…"
