from urllib.parse import quote, urlencode

from pydantic import BaseModel

from virtus.tool import Tool, tool
from virtus.tools.net import Net


class Article(BaseModel):
    title: str
    summary: str
    url: str


def wikipedia(*, net: Net = Net(), count: int = 3, lang: str = "en") -> Tool:
    """The `wikipedia` tool, through `net`; `count` and `lang` are its defaults for the model's arguments."""

    def wikipedia(query: str, count: int = count, lang: str = lang) -> list[Article]:
        """Look a topic up on Wikipedia: the `count` best-matching articles with a summary and a link.
        `lang` picks the edition, e.g. "pt"."""
        base = f"https://{lang}.wikipedia.org"
        search = f"{base}/w/rest.php/v1/search/page?{urlencode({'q': query, 'limit': count})}"
        found = _Search.model_validate_json(net.get(search))
        summaries = (net.get(f"{base}/api/rest_v1/page/summary/{quote(page.key)}") for page in found.pages)
        return [_article(_Summary.model_validate_json(summary)) for summary in summaries]

    return tool(wikipedia)


def _article(summary: "_Summary") -> Article:
    return Article(title=summary.title, summary=summary.extract, url=summary.content_urls.desktop.page)


class _Page(BaseModel):
    key: str


class _Search(BaseModel):
    pages: list[_Page]


class _Urls(BaseModel):
    page: str


class _ContentUrls(BaseModel):
    desktop: _Urls


class _Summary(BaseModel):
    title: str
    extract: str = ""
    content_urls: _ContentUrls
