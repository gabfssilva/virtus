import json
from html.parser import HTMLParser

from virtus.tool import Tool, tool
from virtus.tools.net import Net, clip


def http(*, net: Net = Net(), max_chars: int = 20_000) -> Tool:
    """The `http` tool, through `net`; `max_chars` is how much of a body it returns unless the model asks for more."""

    def http(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str | None = None,
        max_chars: int = max_chars,
    ) -> str:
        """Make an HTTP request. Returns the status, the main headers and the body as text: JSON reindented,
        HTML reduced to its readable text, other binary content described by type and size. Error statuses
        come back the same way, not as failures."""
        encoded = None if body is None else body.encode()
        response = net.request(url, method=method.upper(), headers=headers, body=encoded)
        kind = response.headers.get("content-type", "")
        names = ("content-type", "location")
        shown = (f"{name}: {response.headers[name]}" for name in names if name in response.headers)
        return "\n".join((f"HTTP {response.status}", *shown, "", clip(_render(response.body, kind), max_chars)))

    return tool(http)


def _render(body: bytes, kind: str) -> str:
    media, *params = (part.strip() for part in kind.split(";"))
    media = media.lower()
    charset = next((p.removeprefix("charset=") for p in params if p.startswith("charset=")), "utf-8")

    if "json" in media:
        try:
            return json.dumps(json.loads(body), indent=1, ensure_ascii=False)
        except ValueError:
            return body.decode(charset, errors="replace")
    if media == "text/html" or media == "application/xhtml+xml":
        return _Text.of(body.decode(charset, errors="replace"))
    if media.startswith("text/") or media in {"application/xml", "application/javascript"} or not media:
        return body.decode(charset, errors="replace")
    return f"[{media}, {len(body)} bytes]"


class _Text(HTMLParser):
    """The readable text of a page: block elements become line breaks, scripts and styles vanish."""

    SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head"})
    BLOCK = frozenset(
        {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "header"}
        | {"footer", "nav", "ul", "ol", "table", "pre", "blockquote", "hr"}
    )

    @classmethod
    def of(cls, html: str) -> str:
        parser = cls()
        parser.feed(html)
        lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
        text = "\n".join(lines)
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip()

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.hidden += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self.hidden = max(0, self.hidden - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)
