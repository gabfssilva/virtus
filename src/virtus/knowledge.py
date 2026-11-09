"""Documents on disk the agent can consult, retrieved in the cheapest way their volume allows.

While everything fits the token budget, it all goes into the system prompt. Past that, only an
index of ids and titles goes in and the agent reads entries on demand. Past what the index
allows, the agent searches first and then reads: BM25 with no dependencies, or by meaning when
an `Embedder` is given. Search and reading go by passages, so a long document (a PDF, say) is
found by the part that matters and read a part at a time.

`knowledge(...)` is read-only material the developer provides; `memory(...)` is the same store
with tools for the agent to write and delete entries, so it persists across conversations.
"""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, Unpack

from pydantic import TypeAdapter

from virtus.message import Event
from virtus.middleware import Middleware, Next
from virtus.pdfs import Reader
from virtus.provider import Embedder
from virtus.request import Request
from virtus.tool import Tool, tool


@dataclass(frozen=True)
class Document:
    id: str
    """Path relative to the root it was found under, e.g. `guides/setup.md`."""
    title: str
    content: str
    path: Path


@dataclass(frozen=True)
class Passage:
    """Part of a document, at most the store's `passage` characters long and overlapping the next by `overlap`."""

    id: str
    """The document's id, with `#n` after it when the document has more than one passage."""
    document: Document
    number: int
    """1-based, out of `total`."""
    total: int
    content: str


class Store:
    """Markdown, text and PDF files under `roots`, read again on every access so edits show up at once.

    A `.md` file may start with a frontmatter block of `key: value` lines between `---` fences;
    its `title` names the document. Otherwise the first heading does, and failing that the file name.
    A PDF is its pages' text (OCR where a page has none, extra `pdf`), extracted once per version of the file.

    Search and reading go by passages of at most `passage` characters, cut where a paragraph or else a word
    ends; each starts `overlap` characters before the previous one ended, so a sentence across a cut is whole
    in one of them.

    A PDF's text comes from `reader`; the last `cache` PDFs read are kept, each until its file changes.

    With an `embedder`, `search` ranks by meaning. The vectors are kept in a file named `index` in each
    root's directory, keyed by a hash of the passage, so only new or changed passages are embedded again;
    vectors of another size than the embedder now produces are embedded again too. Without one, it ranks by
    BM25 with `k1` and `b`.
    """

    def __init__(
        self,
        *roots: Path,
        suffixes: Sequence[str] = (".md", ".txt", ".pdf"),
        embedder: Embedder | None = None,
        passage: int = 2_000,
        overlap: int = 200,
        index: str = ".embeddings.json",
        reader: Reader = Reader(),
        cache: int = 256,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not 0 <= overlap < passage // 2:
            raise ValueError(f"overlap must be at least 0 and under half the passage, got {overlap} of {passage}")
        self.roots = roots
        self.suffixes = tuple(suffixes)
        self.embedder = embedder
        self.passage = passage
        self.overlap = overlap
        self.index = index
        self.k1 = k1
        self.b = b

        def extract(file: Path, modified: int, size: int) -> str:
            """Keyed by modification time and size too, so a changed file is extracted again."""
            pages = reader.pages(file.read_bytes())
            return "\n\n".join(f"[page {n}]\n{text}" for n, text in enumerate(pages, 1))

        self._pdf = lru_cache(maxsize=cache)(extract)

    def documents(self) -> tuple[Document, ...]:
        def under(root: Path) -> Iterator[Document]:
            if root.is_file():
                yield self.load(root, root.name)
            elif root.is_dir():
                for file in sorted(root.rglob("*")):
                    if file.suffix in self.suffixes:
                        try:
                            yield self.load(file, str(file.relative_to(root)))
                        except (FileNotFoundError, IsADirectoryError):
                            continue  # deleted, or a directory named like a file, while listing

        return tuple(d for root in self.roots for d in under(root))

    def passages(self) -> tuple[Passage, ...]:
        return tuple(p for d in self.documents() for p in _passages(d, self.passage, self.overlap))

    def read(self, id: str) -> Document | None:
        return next((d for d in self.documents() if d.id == id), None)

    def load(self, file: Path, id: str) -> Document:
        """A heading that names a markdown document is not repeated in its content; a PDF's pages are marked."""
        if file.suffix == ".pdf":
            stat = file.stat()
            return Document(id, file.stem, self._pdf(file, stat.st_mtime_ns, stat.st_size), file)
        if file.suffix != ".md":
            return Document(id, file.stem, file.read_text().strip(), file)

        frontmatter, body = _split(file.read_text())
        if title := frontmatter.get("title"):
            return Document(id, title, body.strip(), file)
        if heading := re.search(r"^#\s+(.+)$", body, re.MULTILINE):
            return Document(
                id, heading.group(1).strip(), (body[: heading.start()] + body[heading.end() :]).strip(), file
            )
        return Document(id, file.stem, body.strip(), file)

    async def search(self, query: str, limit: int = 5) -> list[Passage]:
        """The `limit` passages nearest the query: by cosine with an embedder, else by BM25 over title and text."""
        passages = self.passages()
        if self.embedder is None:
            return _bm25(passages, query, limit, self.k1, self.b)

        [vector] = await self.embedder.embed([query], "query")
        vectors = [v for root in self.roots for v in await self._vectors(root, len(vector))]
        ranked = sorted(zip(map(lambda v: _cosine(vector, v), vectors), passages, strict=True), key=lambda p: -p[0])
        return [passage for _, passage in ranked[:limit]]

    async def _vectors(self, root: Path, size: int) -> list[list[float]]:
        """The vectors of the passages under `root`, from its index where the text hasn't changed."""
        assert self.embedder is not None
        passages = [p for p in self.passages() if p.document.path == root or p.document.path.is_relative_to(root)]
        index = (root if root.is_dir() else root.parent) / self.index
        cached = _INDEX.validate_json(index.read_bytes()) if index.exists() else {}
        keys = [_digest(p) for p in passages]
        kept = {k: v for k, v in cached.items() if k in keys and len(v) == size}

        missing = [(k, p) for k, p in zip(keys, passages, strict=True) if k not in kept]
        if missing:
            fresh = await self.embedder.embed([_text(p) for _, p in missing], "document")
            kept |= {k: v for (k, _), v in zip(missing, fresh, strict=True)}
        if kept != cached:
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps(kept))
        return [kept[k] for k in keys]


class Options(TypedDict, total=False):
    """The `Store` settings `knowledge` and `memory` pass on."""

    suffixes: Sequence[str]
    embedder: Embedder | None
    passage: int
    overlap: int
    index: str
    reader: Reader
    cache: int
    k1: float
    b: float


KNOWLEDGE = "Reference material you can rely on."
MEMORY = "Notes you kept from earlier conversations. Save new ones with {name}_remember."


def knowledge(
    *paths: str | Path,
    name: str = "knowledge",
    intro: str = KNOWLEDGE,
    max_tokens: int = 4_000,
    chars_per_token: int = 4,
    hits: int = 5,
    snippet: int = 160,
    **options: Unpack[Options],
) -> Middleware:
    """Let the agent consult `.md`, `.txt` and `.pdf` files, or whole directories of them, read-only.

    `name` prefixes the tools and the section in the system prompt, so two knowledge bases can
    coexist; `intro` opens that section. `max_tokens` (estimated as `chars_per_token` characters each)
    is how much of the prompt the content, or failing that the index, may take before the agent has to
    read or search instead. A search lists `hits` passages, each with its first `snippet` characters.
    `options` go to the `Store`: an `embedder` makes the search semantic, `passage` and `overlap` size
    the parts it goes by.
    """
    return _consult(
        Store(*map(Path, paths), **options),
        name,
        intro,
        max_tokens,
        chars_per_token,
        hits,
        snippet,
        manage=(),
    )


def memory(
    path: str | Path,
    *,
    name: str = "memory",
    intro: str = MEMORY,
    max_tokens: int = 4_000,
    chars_per_token: int = 4,
    hits: int = 5,
    snippet: int = 160,
    **options: Unpack[Options],
) -> Middleware:
    """Give the agent a directory of markdown notes it writes and reads back in later conversations.

    Each note is a file named after its title, with the title and creation time in frontmatter.
    Notes are plain files, so they can be read and edited by hand. `{name}` in `intro` is the name;
    the other arguments are those of `knowledge`.
    """
    root = Path(path)
    store = Store(root, **options)

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
        intro.format(name=name),
        max_tokens,
        chars_per_token,
        hits,
        snippet,
        manage=(tool(remember, name=f"{name}_remember"), tool(forget, name=f"{name}_forget")),
    )


def _consult(
    store: Store,
    name: str,
    intro: str,
    max_tokens: int,
    chars_per_token: int,
    hits: int,
    snippet: int,
    *,
    manage: tuple[Tool, ...],
) -> Middleware:
    def read(id: str) -> str:
        """A long entry is read a passage at a time: its id alone gives the first, `id#n` the n-th."""
        passages = {p.id: p for p in store.passages()}
        if (passage := passages.get(id) or passages.get(f"{id}#1")) is None:
            raise FileNotFoundError(f"no entry {id!r}")
        text = f"# {passage.document.title}\n\n{passage.content}"
        if passage.total == 1:
            return text
        following = f"; next is {passage.document.id}#{passage.number + 1}" if passage.number < passage.total else ""
        return f"{text}\n\n[passage {passage.number} of {passage.total}{following}]"

    async def search(query: str) -> str:
        found = await store.search(query, hits)
        lines = (f"- {p.id}: {p.document.title}\n  {_snippet(p.content, snippet)}" for p in found)
        return "\n".join(lines) or "No matches."

    how = "by meaning" if store.embedder else "by keywords"
    reading = f"Read one entry of {name} by id; a long one a passage (`id#n`) at a time."
    reader = tool(read, name=f"{name}_read", description=reading)
    searcher = tool(search, name=f"{name}_search", description=f"Search {name} {how}; returns ids to read.")

    async def consult(request: Request, next: Next) -> AsyncIterator[Event]:
        documents = store.documents()
        full = "\n\n".join(f"### {d.title} ({d.id})\n{d.content}" for d in documents)
        index = "\n".join(f"- {d.id}: {d.title}" for d in documents)

        if len(full) // chars_per_token <= max_tokens:
            body, tools = full or "(nothing yet)", manage
        elif len(index) // chars_per_token <= max_tokens:
            body, tools = f"{index}\n\nRead an entry with {name}_read.", (reader, *manage)
        else:
            body = f"{len(documents)} entries. Find the relevant ones with {name}_search, then {name}_read them."
            tools = (searcher, reader, *manage)

        section = f"## {name}\n\n{intro}\n\n{body}"
        system = "\n\n".join(part for part in (request.system, section) if part)

        async for event in next(replace(request, system=system, tools=(*request.tools, *tools))):
            yield event

    return consult


def _passages(document: Document, size: int, overlap: int) -> list[Passage]:
    """Cut where a paragraph, or else a word, ends in the second half of the window; the next passage
    starts at a word `overlap` characters before the cut."""
    text, cuts, start = document.content, [], 0
    while len(text) - start > size:
        window = text[start : start + size]
        cut = window.rfind("\n\n", size // 2)
        cut = cut if cut != -1 else window.rfind(" ", size // 2)
        end = start + (cut if cut != -1 else size)
        cuts.append((start, end))
        space = text.find(" ", end - overlap, end)
        start = space + 1 if space != -1 else end - overlap
    cuts.append((start, len(text)))
    total = len(cuts)
    return [
        Passage(document.id if total == 1 else f"{document.id}#{n}", document, n, total, text[a:b].strip())
        for n, (a, b) in enumerate(cuts, 1)
    ]


def _text(passage: Passage) -> str:
    return f"{passage.document.title}\n\n{passage.content}"


def _bm25(passages: Sequence[Passage], query: str, limit: int, k1: float, b: float) -> list[Passage]:
    """Rank passages by BM25 over title and text; only positive scores count."""
    bags = [Counter(_tokens(_text(p))) for p in passages]
    total = len(passages)
    average = sum(sum(bag.values()) for bag in bags) / total if total else 0.0

    def idf(term: str) -> float:
        containing = sum(term in bag for bag in bags)
        return math.log(1 + (total - containing + 0.5) / (containing + 0.5))

    def score(bag: Counter[str]) -> float:
        length = sum(bag.values())
        return sum(
            idf(term) * bag[term] * (k1 + 1) / (bag[term] + k1 * (1 - b + b * length / average))
            for term in set(_tokens(query))
            if term in bag
        )

    ranked = sorted(zip(map(score, bags), passages, strict=True), key=lambda pair: -pair[0])
    return [passage for value, passage in ranked[:limit] if value > 0]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norms if norms else 0.0


def _digest(passage: Passage) -> str:
    return hashlib.sha256(_text(passage).encode()).hexdigest()


_INDEX = TypeAdapter(dict[str, list[float]])

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


def _snippet(content: str, length: int) -> str:
    collapsed = " ".join(content.split())
    return collapsed if len(collapsed) <= length else f"{collapsed[:length].rstrip()}…"
