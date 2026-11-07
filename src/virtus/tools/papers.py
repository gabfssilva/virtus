"""Scholarly search: `arxiv` for preprints, `openalex` for the published record. Both keyless; both answer `Paper`s."""

import os
from urllib.parse import urlencode
from xml.etree import ElementTree

from pydantic import BaseModel

from virtus.tool import tool
from virtus.tools.net import get

ATOM = {"a": "http://www.w3.org/2005/Atom"}


class Paper(BaseModel):
    title: str
    authors: list[str]
    year: int | None
    url: str
    citations: int | None = None
    summary: str


@tool
def arxiv(query: str, count: int = 5) -> list[Paper]:
    """Search arXiv preprints by keywords; every word must match."""
    terms = " AND ".join(f"all:{word}" for word in query.split())  # a bare phrase matches on its first word only
    params = {"search_query": terms, "max_results": count, "sortBy": "relevance"}
    feed = ElementTree.fromstring(get(f"https://export.arxiv.org/api/query?{urlencode(params)}"))
    return [
        Paper(
            title=" ".join(entry.findtext("a:title", "", ATOM).split()),
            authors=[name.text or "" for name in entry.findall("a:author/a:name", ATOM)],
            year=int(entry.findtext("a:published", "0", ATOM)[:4]) or None,
            url=entry.findtext("a:id", "", ATOM),
            summary=" ".join(entry.findtext("a:summary", "", ATOM).split()),
        )
        for entry in feed.findall("a:entry", ATOM)
    ]


@tool
def openalex(query: str, count: int = 5) -> list[Paper]:
    """Search published papers across journals and conferences, with citation counts."""
    params = {
        "search": query,
        "per-page": count,
        "select": "id,doi,title,publication_year,cited_by_count,authorships,abstract_inverted_index",
    }
    if key := os.environ.get("OPENALEX_API_KEY"):  # without one, a daily budget is shared by the whole IP address
        params["api_key"] = key
    works = _Works.model_validate_json(get(f"https://api.openalex.org/works?{urlencode(params)}")).results
    return [
        Paper(
            title=work.title or "(untitled)",
            authors=[a.author.display_name for a in work.authorships],
            year=work.publication_year,
            url=work.doi or work.id,
            citations=work.cited_by_count,
            summary=_abstract(work.abstract_inverted_index),
        )
        for work in works
    ]


class _Author(BaseModel):
    display_name: str


class _Authorship(BaseModel):
    author: _Author


class _Work(BaseModel):
    id: str
    title: str | None = None
    publication_year: int | None = None
    cited_by_count: int | None = None
    doi: str | None = None
    authorships: list[_Authorship] = []
    abstract_inverted_index: dict[str, list[int]] | None = None


class _Works(BaseModel):
    results: list[_Work]


def _abstract(index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as word -> positions; put the words back in order."""
    words = sorted((position, word) for word, positions in (index or {}).items() for position in positions)
    return " ".join(word for _, word in words) or "(no abstract)"
