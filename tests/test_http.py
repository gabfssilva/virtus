import importlib
import json
from contextlib import contextmanager
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from virtus import ToolCall
from virtus.tools import http

net = importlib.import_module("virtus.tools.net")


class Server:
    """Stands in for `urlopen`: records requests; answers each with (status, content type, body) or raises the error."""

    def __init__(self, *answers: tuple[int, str, bytes] | HTTPError) -> None:
        self.answers = list(answers)
        self.requests: list[Request] = []

    @contextmanager
    def __call__(self, request: Request, timeout: float):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, HTTPError):
            raise answer
        status, kind, body = answer
        yield _Reply(status, kind, body)


class _Reply(BytesIO):
    def __init__(self, status: int, kind: str, body: bytes) -> None:
        super().__init__(body)
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = kind


def describe_http():
    async def it_sends_method_headers_and_body_and_reindents_json(monkeypatch: pytest.MonkeyPatch):
        server = Server((201, "application/json", b'{"id":7,"tags":["a","b"]}'))
        monkeypatch.setattr(net, "urlopen", server)

        result = await http().execute(
            ToolCall(
                "c",
                "http",
                json.dumps(
                    {
                        "url": "https://api.example.com/items",
                        "method": "post",
                        "headers": {"Authorization": "Bearer t"},
                        "body": '{"name": "x"}',
                    }
                ),
            )
        )

        [request] = server.requests
        assert (request.get_method(), request.full_url) == ("POST", "https://api.example.com/items")
        assert request.data == b'{"name": "x"}'
        assert request.get_header("Authorization") == "Bearer t"
        assert request.get_header("User-agent") == net.AGENT
        body = '{\n "id": 7,\n "tags": [\n  "a",\n  "b"\n ]\n}'
        assert result.content == f"HTTP 201\ncontent-type: application/json\n\n{body}"

    async def it_reduces_html_to_its_text(monkeypatch: pytest.MonkeyPatch):
        page = b"""<html><head><title>T</title><style>p{}</style></head>
        <body><nav>menu</nav><h1>Hello   world</h1><p>First <b>bold</b> line.</p><script>x()</script>
        <ul><li>one</li><li>two</li></ul></body></html>"""
        monkeypatch.setattr(net, "urlopen", Server((200, "text/html; charset=utf-8", page)))

        result = await http().execute(ToolCall("c", "http", json.dumps({"url": "https://example.com"})))

        text = "menu\n\nHello world\n\nFirst bold line.\n\none\n\ntwo"
        assert result.content == f"HTTP 200\ncontent-type: text/html; charset=utf-8\n\n{text}"

    async def it_returns_error_statuses_as_responses(monkeypatch: pytest.MonkeyPatch):
        headers = Message()
        headers["Content-Type"] = "text/plain"
        error = HTTPError("https://example.com/x", 404, "Not Found", headers, BytesIO(b"gone"))
        monkeypatch.setattr(net, "urlopen", Server(error))

        result = await http().execute(ToolCall("c", "http", json.dumps({"url": "https://example.com/x"})))

        assert not result.is_error
        assert result.content == "HTTP 404\ncontent-type: text/plain\n\ngone"

    async def it_describes_binary_bodies_and_truncates_long_ones(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(net, "urlopen", Server((200, "image/png", b"\x89PNG" * 10), (200, "text/plain", b"x" * 30)))

        image = await http().execute(ToolCall("c", "http", json.dumps({"url": "https://example.com/i.png"})))
        text = await http().execute(
            ToolCall("c", "http", json.dumps({"url": "https://example.com/t", "max_chars": 10}))
        )

        assert image.content.endswith("[image/png, 40 bytes]")
        assert text.content.endswith("x" * 10 + "\n… [truncated; 30 chars in total]")
