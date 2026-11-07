from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from virtus.tool import tool


@tool
def websearch(
    query: str,
    kind: Literal["text", "news", "images", "videos", "books"] = "text",
    count: int = 10,
    period: Literal["d", "w", "m", "y"] | None = None,
) -> str:
    """Search the web: pages (`text`), `news`, `images`, `videos` or `books`.

    `period` keeps only the last day, week, month or year. Returns the top `count` results, each as a
    title, its url and what else is known about it.
    """
    from ddgs import DDGS  # the `websearch` extra; a metasearch over Bing, Google, DuckDuckGo and others, no key needed
    from ddgs.exceptions import DDGSException

    engine = DDGS()
    hits: Sequence[Hit]
    try:
        match kind:
            case "text":
                hits = _PAGES.validate_python(engine.text(query, max_results=count, timelimit=period))
            case "news":
                hits = _NEWS.validate_python(engine.news(query, max_results=count, timelimit=period))
            case "images":
                hits = _IMAGES.validate_python(engine.images(query, max_results=count, timelimit=period))
            case "videos":
                hits = _VIDEOS.validate_python(engine.videos(query, max_results=count, timelimit=period))
            case "books":
                hits = _BOOKS.validate_python(engine.books(query, max_results=count, timelimit=period))
    except DDGSException as error:
        if str(error).startswith("No results"):
            return "No results."
        raise
    return "\n\n".join(hit.render() for hit in hits) or "No results."


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
