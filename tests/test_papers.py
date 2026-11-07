import importlib
import json
from contextlib import contextmanager
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from virtus import ToolCall
from virtus.tools import Paper, arxiv, openalex

net = importlib.import_module("virtus.tools.net")

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.02191v1</id>
    <title>On LLM-Enhanced
  Mixed-Type Data Imputation</title>
    <summary>Missing data imputation
  is crucial.</summary>
    <published>2025-01-04T05:05:44Z</published>
    <author><name>Jianwei Wang</name></author>
    <author><name>Kai Wang</name></author>
  </entry>
</feed>"""

WORKS = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1/x",
            "title": "FILLM",
            "publication_year": 2025,
            "cited_by_count": 3,
            "authorships": [{"author": {"display_name": "Subham Jha"}}],
            "abstract_inverted_index": {"imputation": [1], "LLM": [0], "works": [2]},
        },
        {"id": "https://openalex.org/W2", "title": None, "authorships": []},
    ]
}


class Fetch:
    """Stands in for `urlopen`: records urls, answers each from a queue, raising the entries that are exceptions."""

    def __init__(self, *answers: bytes | Exception) -> None:
        self.answers = list(answers)
        self.urls: list[str] = []

    @contextmanager
    def __call__(self, request: Request, timeout: float):
        self.urls.append(request.full_url)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        yield BytesIO(answer)


def describe_arxiv():
    async def it_requires_every_word_and_reads_the_feed(monkeypatch: pytest.MonkeyPatch):
        fetch = Fetch(FEED)
        monkeypatch.setattr(net, "urlopen", fetch)

        result = await arxiv.execute(ToolCall("1", "arxiv", '{"query": "LLM imputation", "count": 2}'))

        assert fetch.urls == [
            "https://export.arxiv.org/api/query?search_query=all%3ALLM+AND+all%3Aimputation&max_results=2&sortBy=relevance"
        ]
        assert json.loads(result.content) == [
            Paper(
                title="On LLM-Enhanced Mixed-Type Data Imputation",
                authors=["Jianwei Wang", "Kai Wang"],
                year=2025,
                url="http://arxiv.org/abs/2501.02191v1",
                summary="Missing data imputation is crucial.",
            ).model_dump()
        ]

    async def it_retries_once_after_a_pause_when_rate_limited(monkeypatch: pytest.MonkeyPatch):
        limited = HTTPError("u", 429, "Too Many Requests", Message(), None)
        fetch = Fetch(limited, FEED)
        paused: list[float] = []
        monkeypatch.setattr(net, "urlopen", fetch)
        monkeypatch.setattr(net.time, "sleep", paused.append)

        result = await arxiv.execute(ToolCall("1", "arxiv", '{"query": "LLM"}'))

        assert len(fetch.urls) == 2
        assert paused == [3]
        assert not result.is_error

    async def it_gives_up_on_other_errors(monkeypatch: pytest.MonkeyPatch):
        fetch = Fetch(HTTPError("u", 500, "Server Error", Message(), None))
        monkeypatch.setattr(net, "urlopen", fetch)

        result = await arxiv.execute(ToolCall("1", "arxiv", '{"query": "LLM"}'))

        assert result.is_error
        assert "500" in result.content


def describe_openalex():
    async def it_rebuilds_abstracts_and_sends_the_key_when_set(monkeypatch: pytest.MonkeyPatch):
        fetch = Fetch(json.dumps(WORKS).encode())
        monkeypatch.setattr(net, "urlopen", fetch)
        monkeypatch.setenv("OPENALEX_API_KEY", "oa-key")

        result = await openalex.execute(ToolCall("1", "openalex", '{"query": "LLM imputation"}'))

        [url] = fetch.urls
        assert url.startswith("https://api.openalex.org/works?search=LLM+imputation&per-page=5&select=")
        assert url.endswith("&api_key=oa-key")
        first, second = json.loads(result.content)
        assert first == Paper(
            title="FILLM",
            authors=["Subham Jha"],
            year=2025,
            url="https://doi.org/10.1/x",
            citations=3,
            summary="LLM imputation works",
        ).model_dump()
        assert (second["title"], second["url"]) == ("(untitled)", "https://openalex.org/W2")
        assert second["summary"] == "(no abstract)"

    async def it_goes_without_a_key(monkeypatch: pytest.MonkeyPatch):
        fetch = Fetch(json.dumps({"results": []}).encode())
        monkeypatch.setattr(net, "urlopen", fetch)
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

        result = await openalex.execute(ToolCall("1", "openalex", '{"query": "x"}'))

        assert "api_key" not in fetch.urls[0]
        assert result.content == "[]"
