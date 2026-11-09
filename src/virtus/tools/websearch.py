from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from virtus.tool import Tool, tool


def websearch(
    *,
    count: int = 10,
    backend: str = "auto",
    region: str = "us-en",
    safesearch: Literal["on", "moderate", "off"] = "moderate",
    timeout: int = 5,
    proxy: str | None = None,
) -> Tool:
    """The `websearch` tool over ddgs, a metasearch with no key: `backend` picks the engines (e.g. "google, bing"),
    and it, `region`, `safesearch`, `timeout` (seconds) and `proxy` go to ddgs as they are. `count` is the
    default for the model's argument."""

    def websearch(
        query: str,
        kind: Literal["text", "news", "images", "videos", "books"] = "text",
        count: int = count,
        period: Literal["d", "w", "m", "y"] | None = None,
    ) -> str:
        """Search the web: pages (`text`), `news`, `images`, `videos` or `books`.

        `period` keeps only the last day, week, month or year. Returns the top `count` results, each as a
        title, its url and what else is known about it.
        """
        from ddgs import DDGS  # the `websearch` extra
        from ddgs.exceptions import DDGSException

        engine = DDGS(proxy=proxy, timeout=timeout)
        search = {"text": engine.text, "news": engine.news, "images": engine.images}
        search |= {"videos": engine.videos, "books": engine.books}
        adapter = {"text": _PAGES, "news": _NEWS, "images": _IMAGES, "videos": _VIDEOS, "books": _BOOKS}[kind]
        try:
            rows = search[kind](
                query, region=region, safesearch=safesearch, timelimit=period, max_results=count, backend=backend
            )
            hits: Sequence[Hit] = adapter.validate_python(rows)
        except DDGSException as error:
            if str(error).startswith("No results"):
                return "No results."
            raise
        return "\n\n".join(hit.render() for hit in hits) or "No results."

    return tool(websearch)


class Hit(BaseModel):
    title: str = ""

    def render(self) -> str:
        return "\n".join(line for line in self.lines() if line)

    def lines(self) -> tuple[str, ...]:
        return (self.title,)


class Page(Hit):
    href: str
    body: str = ""

    def lines(self) -> tuple[str, ...]:
        return (self.title, self.href, self.body)


class News(Hit):
    url: str
    body: str = ""
    date: str = ""
    source: str = ""

    def lines(self) -> tuple[str, ...]:
        return (self.title, self.url, " · ".join(part for part in (self.date[:10], self.source) if part), self.body)


class Image(Hit):
    image: str
    url: str = ""
    source: str = ""
    width: int | None = None
    height: int | None = None

    def lines(self) -> tuple[str, ...]:
        size = f"{self.width}x{self.height}" if self.width and self.height else ""
        return (self.title, self.image, " · ".join(part for part in (size, self.source) if part), self.url)


class Video(Hit):
    content: str
    description: str = ""
    publisher: str = ""
    uploader: str = ""
    duration: str = ""
    published: str = ""

    def lines(self) -> tuple[str, ...]:
        details = (self.published[:10], self.publisher or self.uploader, self.duration)
        about = " · ".join(part for part in details if part)
        return (self.title, self.content, about, self.description)


class Book(Hit):
    url: str = ""
    author: str = ""
    publisher: str = ""
    info: str = ""

    def lines(self) -> tuple[str, ...]:
        return (self.title, self.url, " · ".join(part for part in (self.author, self.publisher, self.info) if part))


_PAGES = TypeAdapter(list[Page])
_NEWS = TypeAdapter(list[News])
_IMAGES = TypeAdapter(list[Image])
_VIDEOS = TypeAdapter(list[Video])
_BOOKS = TypeAdapter(list[Book])
