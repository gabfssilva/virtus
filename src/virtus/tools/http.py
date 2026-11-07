import json
from html.parser import HTMLParser

from virtus.tool import tool
from virtus.tools import net


@tool
def http(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    max_chars: int = 20_000,
) -> str:
    """Make an HTTP request. Returns the status, the main headers and the body as text: JSON reindented,
    HTML reduced to its readable text, other binary content described by type and size. Error statuses
    come back the same way, not as failures."""
    response = net.request(url, method=method.upper(), headers=headers, body=None if body is None else body.encode())
    kind = response.headers.get("content-type", "")
    shown = (f"{name}: {response.headers[name]}" for name in ("content-type", "location") if name in response.headers)
    return "\n".join((f"HTTP {response.status}", *shown, "", _clip(_render(response.body, kind), max_chars)))


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


def _clip(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else f"{text[:max_chars]}\n… [truncated; {len(text)} chars in total]"


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
