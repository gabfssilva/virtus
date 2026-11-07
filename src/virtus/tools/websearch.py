from pydantic import BaseModel, TypeAdapter

from virtus.tool import tool


class Hit(BaseModel):
    title: str
    href: str
    body: str = ""


_hits = TypeAdapter(list[Hit])


@tool
def websearch(query: str, count: int = 10) -> str:
    """Search the web. Returns the top `count` results, each as a title, its url and a short summary."""
    from ddgs import DDGS  # the `websearch` extra; a metasearch over Bing, Google, DuckDuckGo and others, no key needed

    hits = _hits.validate_python(DDGS().text(query, max_results=count))
    return "\n\n".join(f"{hit.title}\n{hit.href}\n{hit.body}" for hit in hits) or "No results."
