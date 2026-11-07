import pytest

from virtus import ToolCall, ToolResult
from virtus.tools import websearch

ENGINE = "ddgs.ddgs.DDGS.text"  # `ddgs.DDGS` is a proxy; the class that answers lives one module down


def describe_websearch():
    async def it_searches_and_lists_the_hits(monkeypatch: pytest.MonkeyPatch):
        asked: list[tuple[str, int]] = []

        def text(self: object, query: str, max_results: int) -> list[dict[str, str]]:
            asked.append((query, max_results))
            return [
                {
                    "title": "Coroutines and tasks",
                    "href": "https://docs.python.org/3/library/asyncio-task.html",
                    "body": "Tasks.",
                },
                {"title": "TaskGroup", "href": "https://example.org/taskgroup"},
            ]

        monkeypatch.setattr(ENGINE, text)

        result = await websearch.execute(ToolCall("1", "websearch", '{"query": "asyncio TaskGroup", "count": 3}'))

        assert asked == [("asyncio TaskGroup", 3)]
        assert result == ToolResult(
            "1",
            "websearch",
            "Coroutines and tasks\nhttps://docs.python.org/3/library/asyncio-task.html\nTasks."
            "\n\nTaskGroup\nhttps://example.org/taskgroup\n",
        )

    async def it_asks_for_ten_results_by_default_and_says_when_there_are_none(monkeypatch: pytest.MonkeyPatch):
        asked: list[int] = []
        monkeypatch.setattr(ENGINE, lambda self, query, max_results: asked.append(max_results) or [])

        result = await websearch.execute(ToolCall("1", "websearch", '{"query": "qzxv"}'))

        assert asked == [10]
        assert result.content == "No results."

    def it_describes_itself_for_the_model():
        assert websearch.name == "websearch"
        assert websearch.description.startswith("Search the web.")
        assert websearch.schema["required"] == ["query"]
        assert websearch.schema["properties"] == {
            "query": {"type": "string", "title": "Query"},
            "count": {"type": "integer", "title": "Count", "default": 10},
        }
