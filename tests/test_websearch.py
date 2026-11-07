import json

import pytest

from virtus import ToolCall, ToolResult
from virtus.tools import websearch

ENGINE = "ddgs.ddgs.DDGS"  # `ddgs.DDGS` is a proxy; the class that answers lives one module down


def _answering(rows: list[dict[str, object]], asked: list[tuple[str, dict[str, object]]]):
    def search(self: object, query: str, **options: object) -> list[dict[str, object]]:
        asked.append((query, options))
        return rows

    return search


def describe_websearch():
    async def it_searches_pages_and_lists_the_hits(monkeypatch: pytest.MonkeyPatch):
        asked: list[tuple[str, dict[str, object]]] = []
        rows: list[dict[str, object]] = [
            {"title": "Coroutines and tasks", "href": "https://docs.python.org/3/library/asyncio-task", "body": "A."},
            {"title": "TaskGroup", "href": "https://example.org/taskgroup"},
        ]
        monkeypatch.setattr(f"{ENGINE}.text", _answering(rows, asked))

        result = await websearch.execute(ToolCall("1", "websearch", '{"query": "asyncio TaskGroup", "count": 3}'))

        assert asked == [("asyncio TaskGroup", {"max_results": 3, "timelimit": None})]
        assert result == ToolResult(
            "1",
            "websearch",
            "Coroutines and tasks\nhttps://docs.python.org/3/library/asyncio-task\nA."
            "\n\nTaskGroup\nhttps://example.org/taskgroup",
        )

    async def it_searches_news_within_a_period(monkeypatch: pytest.MonkeyPatch):
        asked: list[tuple[str, dict[str, object]]] = []
        rows: list[dict[str, object]] = [
            {
                "title": "Python 3.15 released",
                "url": "https://example.org/py315",
                "body": "The new release.",
                "date": "2026-10-07T09:00:00+00:00",
                "source": "Example News",
                "image": "https://example.org/i.png",
            }
        ]
        monkeypatch.setattr(f"{ENGINE}.news", _answering(rows, asked))

        call = ToolCall("1", "websearch", json.dumps({"query": "python", "kind": "news", "period": "w"}))
        result = await websearch.execute(call)

        assert asked == [("python", {"max_results": 10, "timelimit": "w"})]
        expected = "Python 3.15 released\nhttps://example.org/py315\n2026-10-07 · Example News\nThe new release."
        assert result.content == expected

    async def it_renders_images_videos_and_books_by_their_own_fields(monkeypatch: pytest.MonkeyPatch):
        image = {"title": "A cat", "image": "https://example.org/cat.jpg", "url": "https://example.org/cats"}
        images: list[dict[str, object]] = [{**image, "width": 800, "height": 600, "source": "Bing", "thumbnail": "t"}]
        video = {"title": "Talk", "content": "https://example.org/v", "publisher": "YouTube", "duration": "12:03"}
        videos: list[dict[str, object]] = [{**video, "published": "2026-01-02T00:00:00", "description": "A talk."}]
        book = {"title": "Fluent Python", "author": "Luciano Ramalho", "publisher": "O'Reilly", "info": "2022, pdf"}
        books: list[dict[str, object]] = [{**book, "url": "https://example.org/b"}]
        monkeypatch.setattr(f"{ENGINE}.images", _answering(images, []))
        monkeypatch.setattr(f"{ENGINE}.videos", _answering(videos, []))
        monkeypatch.setattr(f"{ENGINE}.books", _answering(books, []))

        async def search(kind: str) -> str:
            call = ToolCall("1", "websearch", json.dumps({"query": "q", "kind": kind}))
            return (await websearch.execute(call)).content

        assert await search("images") == "A cat\nhttps://example.org/cat.jpg\n800x600 · Bing\nhttps://example.org/cats"
        assert await search("videos") == "Talk\nhttps://example.org/v\n2026-01-02 · YouTube · 12:03\nA talk."
        assert await search("books") == "Fluent Python\nhttps://example.org/b\nLuciano Ramalho · O'Reilly · 2022, pdf"

    async def it_says_when_there_are_no_results(monkeypatch: pytest.MonkeyPatch):
        from ddgs.exceptions import DDGSException

        def nothing(self: object, query: str, **options: object) -> list[dict[str, object]]:
            raise DDGSException("No results found.")

        monkeypatch.setattr(f"{ENGINE}.text", _answering([], []))
        monkeypatch.setattr(f"{ENGINE}.books", nothing)

        empty = await websearch.execute(ToolCall("1", "websearch", '{"query": "qzxv"}'))
        failed = await websearch.execute(ToolCall("1", "websearch", '{"query": "qzxv", "kind": "books"}'))

        assert (empty.content, failed.content) == ("No results.", "No results.")

    def it_describes_itself_for_the_model():
        assert websearch.name == "websearch"
        assert websearch.description.startswith("Search the web:")
        assert websearch.schema["required"] == ["query"]
        kinds = ["text", "news", "images", "videos", "books"]
        assert websearch.schema["properties"] == {
            "query": {"type": "string", "title": "Query"},
            "kind": {"type": "string", "title": "Kind", "default": "text", "enum": kinds},
            "count": {"type": "integer", "title": "Count", "default": 10},
            "period": {
                "anyOf": [{"type": "string", "enum": ["d", "w", "m", "y"]}, {"type": "null"}],
                "title": "Period",
                "default": None,
            },
        }
