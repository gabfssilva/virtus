import importlib
import json
from contextlib import contextmanager
from io import BytesIO
from urllib.request import Request

import pytest

from virtus import ToolCall
from virtus.tools import Article, wikipedia

net = importlib.import_module("virtus.tools.net")

SEARCH = json.dumps({"pages": [{"id": 1, "key": "Honey_bee", "title": "Honey bee"}]}).encode()
SUMMARY = json.dumps(
    {
        "title": "Honey bee",
        "extract": "A honey bee is a eusocial flying insect.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Honey_bee"}},
    }
).encode()


class Fetch:
    def __init__(self, *answers: bytes) -> None:
        self.answers = list(answers)
        self.requests: list[Request] = []

    @contextmanager
    def __call__(self, request: Request, timeout: float):
        self.requests.append(request)
        yield BytesIO(self.answers.pop(0))


def describe_wikipedia():
    async def it_searches_then_summarizes_each_page(monkeypatch: pytest.MonkeyPatch):
        fetch = Fetch(SEARCH, SUMMARY)
        monkeypatch.setattr(net, "urlopen", fetch)

        result = await wikipedia().execute(ToolCall("c", "wikipedia", json.dumps({"query": "honey bee", "count": 1})))

        assert [r.full_url for r in fetch.requests] == [
            "https://en.wikipedia.org/w/rest.php/v1/search/page?q=honey+bee&limit=1",
            "https://en.wikipedia.org/api/rest_v1/page/summary/Honey_bee",
        ]
        assert fetch.requests[0].get_header("User-agent") == net.AGENT
        assert json.loads(result.content) == [
            Article(
                title="Honey bee",
                summary="A honey bee is a eusocial flying insect.",
                url="https://en.wikipedia.org/wiki/Honey_bee",
            ).model_dump()
        ]

    async def it_uses_the_requested_edition(monkeypatch: pytest.MonkeyPatch):
        fetch = Fetch(json.dumps({"pages": []}).encode())
        monkeypatch.setattr(net, "urlopen", fetch)

        result = await wikipedia().execute(ToolCall("c", "wikipedia", json.dumps({"query": "abelha", "lang": "pt"})))

        assert fetch.requests[0].full_url.startswith("https://pt.wikipedia.org/w/rest.php/v1/search/page?q=abelha")
        assert result.content == "[]"
